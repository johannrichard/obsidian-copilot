import { BaseCloudDBProvider } from "./BaseCloudDBProvider";
import { Embeddings } from "@langchain/core/embeddings";
import { App, Notice, Platform, requestUrl } from "obsidian";
import { VectorDocument } from "./dbProvider";
import { logError, logInfo } from "@/logger";
import { VECTORIZE_BATCH_LIMIT } from "@/constants";
import { subscribeToSettingsChange, getSettings } from "@/settings/model";
import { getVectorLength } from "./searchUtils";
import EmbeddingsManager from "@/LLMProviders/embeddingManager";
import { CustomError } from "@/error";

interface PointStruct {
  id: string | number;
  vector: number[];
  payload?: Record<string, unknown>;
}

export class QdrantDBProvider extends BaseCloudDBProvider {
  private baseUrl: string;
  private collectionBaseUrl: string;
  private qdrantApiKey: string;

  constructor(app: App) {
    super(app);
    // Subscribe to settings changes
    subscribeToSettingsChange(async () => {
      const settings = getSettings();

      // Handle mobile index loading setting change
      if (Platform.isMobile && settings.disableIndexOnMobile) {
        this.isInitialized = false;
      } else if (Platform.isMobile && !settings.disableIndexOnMobile && !this.isInitialized) {
        // Re-initialize DB if mobile setting is enabled
        this.collectionName =
          settings.qdrantCollectionName && settings.qdrantCollectionName.trim() !== ""
            ? settings.qdrantCollectionName.trim()
            : this.getVaultIdentifier();

        await this.initializeDB(await EmbeddingsManager.getInstance().getEmbeddingsAPI());
      }
    });
  }

  public async initializeDB(embeddingInstance: Embeddings): Promise<void> {
    const settings = getSettings();

    this.isInitializing = true;
    this.embeddingInstance = embeddingInstance;
    this.embeddingModelName = EmbeddingsManager.getModelName(embeddingInstance);
    this.vectorLength = await getVectorLength(embeddingInstance);
    const qdrantHost = settings.qdrantUrl || "";
    this.qdrantApiKey = settings.qdrantApiKey || "";
    this.baseUrl = `${qdrantHost}`; // Adjust as needed
    this.collectionBaseUrl = `${this.baseUrl}/collections/${this.collectionName}`.replace(
      /(?<!http:|https:)\/\//g,
      "/"
    );

    try {
      await this.loadMetadata();
      await this.ensureCollectionExists();
      await this.validateVectorDimensions();
      await this.createPayloadIndex("path"); // Ensure the "path" index exists
      this.isInitialized = true;
      this.isInitializing = false;
      logInfo(
        `Initialized Qdrant provider with model: ${this.embeddingModelName} and collection: ${this.collectionName}`
      );
    } catch (error) {
      this.isInitialized = false;
      this.isInitializing = false;
      logError("Error initializing Qdrant provider:", error);
      new Notice("Failed to initialize Qdrant provider. Will retry when network is available.");
    }
  }

  public async saveDB(): Promise<void> {
    try {
      if (this.pendingDocuments.length === 0) {
        return;
      }

      const points: PointStruct[] = this.pendingDocuments.map((doc) => ({
        id: doc.id,
        vector: doc.embedding || [],
        payload: {
          path: doc.path,
          title: doc.title,
          content: doc.content,
          created_at: doc.created_at,
          ctime: doc.ctime,
          mtime: doc.mtime,
          tags: doc.tags,
          extension: doc.extension,
          nchars: doc.nchars,
          embeddingModel: doc.embeddingModel,
          vault: this.getVaultIdentifier(), // needed to be able to delete points by filter
          metadata: doc.metadata,
        },
      }));

      const url = `${this.collectionBaseUrl}/points`;
      const response = await requestUrl({
        url,
        method: "PUT",
        contentType: "application/json",
        headers: {
          "api-key": this.qdrantApiKey,
        },
        body: JSON.stringify(
          {
            points: points,
          },
          null,
          4
        ),
        throw: false,
      });

      if (response.status < 200 || response.status >= 300) {
        const errorData =
          typeof response.json === "string" ? JSON.parse(response.json) : response.json;
        throw new Error(`Failed to upsert vectors: ${JSON.stringify(errorData)}`);
      }

      this.pendingDocuments = [];
      await this.saveMetadata();
      logInfo(`Saved ${points.length} documents to Qdrant`);
    } catch (error) {
      logError("Error saving to Qdrant:", error);
      throw error;
    }
  }

  public async recreateIndex(embeddingInstance: Embeddings): Promise<void> {
    try {
      if (!this.collectionExists()) {
        await this.createCollection();
      }
      await this.clearCollection();
      this.reinitializeMetadataCache();
      this.isInitialized = true;
      logInfo(`Recreated Qdrant index with model: ${this.embeddingModelName}`);
    } catch (error) {
      logError("Error recreating Qdrant index:", error);
      throw error;
    }
  }

  public getDb(): void | undefined {
    if (!this.isInitialized) {
      console.warn("Database not initialized. Some features may be limited.");
    }

    throw new Error("Method not implemented.");
  }

  protected async deleteFromCloud(id: string): Promise<void> {
    try {
      const url = `${this.collectionBaseUrl}/points/${id}`;
      const response = await requestUrl({
        url,
        method: "DELETE",
        headers: {
          "api-key": `${this.qdrantApiKey}`,
        },
        throw: false,
      });

      if (response.status < 200 || response.status >= 300) {
        const errorData =
          typeof response.json === "string" ? JSON.parse(response.json) : response.json;
        throw new Error(`Failed to delete point ${id}: ${JSON.stringify(errorData)}`);
      }

      logInfo(`Deleted document ${id} from Qdrant`);
    } catch (error) {
      logError(`Error deleting document ${id} from Qdrant:`, error);
      throw error;
    }
  }

  /**
   * PRovider specific method to search by embedding
   * @param embedding
   * @param limit
   * @param threshold
   * @returns
   */
  protected async searchCloudByEmbedding(
    embedding: number[],
    limit: number,
    threshold: number
  ): Promise<VectorDocument[]> {
    try {
      logInfo("Searching Qdrant");
      const url = `${this.collectionBaseUrl}/points/search`;
      const response = await requestUrl({
        url,
        method: "POST",
        contentType: "application/json",
        headers: {
          "api-key": `${this.qdrantApiKey}`,
        },
        body: JSON.stringify(
          {
            vector: embedding,
            limit,
            with_payload: true,
            with_vectors: true,
            score_threshold: threshold,
          },
          null,
          4
        ),
        throw: false,
      });

      if (response.status < 200 || response.status >= 300) {
        const errorData =
          typeof response.json === "string" ? JSON.parse(response.json) : response.json;
        throw new Error(`Failed to search points: ${JSON.stringify(errorData)}`);
      }

      const searchResults = response.json.result;

      return searchResults.map((result: any) => ({
        id: result.id.toString(),
        path: result.payload?.path as string,
        title: result.payload?.title as string,
        content: result.payload?.content as string,
        embedding: result.vector as number[],
        created_at: result.payload?.created_at as number,
        ctime: result.payload?.ctime as number,
        mtime: result.payload?.mtime as number,
        tags: result.payload?.tags as string[],
        extension: result.payload?.extension as string,
        nchars: result.payload?.nchars as number,
        embeddingModel: result.payload?.embeddingModel as string,
        similarity: result.score,
        metadata: result.payload as Record<string, any>,
      }));
    } catch (error) {
      logError("Error searching Qdrant:", error);
      throw error;
    }
  }

  protected async searchCloudByPath(path: string): Promise<VectorDocument[]> {
    try {
      logInfo(`Searching Qdrant for path: ${path}`);
      const url = `${this.collectionBaseUrl}/points/scroll`;
      let offset: string | number | undefined = undefined;
      let hasNextPage = true;
      const results: VectorDocument[] = [];

      while (hasNextPage) {
        const response = await requestUrl({
          url,
          method: "POST",
          contentType: "application/json",
          headers: {
            "api-key": `${this.qdrantApiKey}`,
          },
          body: JSON.stringify({
            offset: offset,
            with_payload: true,
            with_vector: true,
            filter: {
              must: [
                {
                  key: "path",
                  match: {
                    value: path,
                  },
                },
              ],
            },
          }),
          throw: false,
        });

        if (response.status < 200 || response.status >= 300) {
          const errorData =
            typeof response.json === "string" ? JSON.parse(response.json) : response.json;
          throw new Error(`Failed to scroll points: ${JSON.stringify(errorData)}`);
        }

        const scrollResult = response.json.result;
        if (!scrollResult || !scrollResult.points) {
          hasNextPage = false;
          continue;
        }

        const points = scrollResult.points;
        if (points.length === 0) {
          hasNextPage = false;
          continue;
        }

        for (const point of points) {
          results.push({
            id: point.id.toString(),
            path: point.payload?.path as string,
            title: point.payload?.title as string,
            content: point.payload?.content as string,
            embedding: point.vector as number[],
            created_at: point.payload?.created_at as number,
            ctime: point.payload?.ctime as number,
            mtime: point.payload?.mtime as number,
            tags: point.payload?.tags as string[],
            extension: point.payload?.extension as string,
            nchars: point.payload?.nchars as number,
            embeddingModel: point.payload?.embeddingModel as string,
            metadata: point.payload as Record<string, any>,
          });
        }

        offset = scrollResult.next_page_offset;
        hasNextPage = scrollResult.next_page_offset !== null;
      }

      return results;
    } catch (error) {
      logError(`Error searching Qdrant by path ${path}:`, error);
      throw error;
    }
  }

  protected async collectionExists(): Promise<boolean> {
    logInfo(`Check if Qdrant collection ${this.collectionName} exists`);
    try {
      const url = `${this.collectionBaseUrl}/exists`;
      const response = await requestUrl({
        url,
        headers: {
          "api-key": `${this.qdrantApiKey}`,
        },
        throw: false,
      });

      if (response.status < 200 || response.status >= 300) {
        const errorData =
          typeof response.json === "string" ? JSON.parse(response.json) : response.json;
        logError(`Error checking if Qdrant index exists: ${JSON.stringify(errorData)}`);
        return false; // Or throw the error if you prefer to halt execution
      }

      const collection = response.json.result;
      return collection?.exists;
    } catch (error) {
      logError("Error checking if Qdrant index exists:", error);
      return false;
    }
  }

  protected async createCollection(): Promise<void> {
    try {
      const url = this.collectionBaseUrl;
      const response = await requestUrl({
        url,
        method: "PUT",
        contentType: "application/json",
        headers: {
          "api-key": `${this.qdrantApiKey}`,
        },
        body: JSON.stringify({
          vectors: {
            size: this.vectorLength,
            distance: "Cosine",
          },
        }),
        throw: false,
      });
      if (!response.json.result) {
        throw new Error(`Failed to create index: ${JSON.stringify(response.json)}`);
      }
      logInfo(`Created Qdrant index: ${this.collectionName}`);
    } catch (error) {
      logError("Error creating Qdrant index:", error);
    }
  }

  protected async clearCollection(): Promise<void> {
    // Depending on API credentials, deleting the collection is not possible and we must use other means
    try {
      const url = `${this.collectionBaseUrl}/points/delete`;
      await requestUrl({
        url,
        method: "POST",
        headers: {
          "api-key": `${this.qdrantApiKey}`,
        },
        body: JSON.stringify(
          {
            filter: {
              // We delete by clearing the index via a filter
              must: [
                {
                  key: "vault",
                  match: {
                    value: this.getVaultIdentifier(),
                  },
                },
              ],
            },
          },
          null,
          4
        ),
        throw: false,
      });
      logInfo(`Deleted Qdrant index: ${this.collectionName}`);
    } catch (error) {
      logError("Error deleting Qdrant index:", error);
      throw error;
    }
  }

  protected async validateVectorDimensions(): Promise<boolean> {
    try {
      const response = await requestUrl({
        url: this.collectionBaseUrl,
        headers: { "api-key": `${this.qdrantApiKey}` },
        throw: false,
      });
      const collectionInfo = response.json.result;
      if (collectionInfo.config.params.vectors?.size !== this.vectorLength) {
        logInfo(
          `Qdrant index ${this.collectionName} has incorrect dimensions: ${collectionInfo.config.params.vectors?.size}, expected: ${this.vectorLength}`
        );
        return false;
      }
      return true;
    } catch (error) {
      logError("Error validating Qdrant index dimensions:", error);
      return false;
    }
  }

  // ... (Other methods like saveDB, searchCloud, etc., need similar refactoring)

  protected async getDocsFromCloudById(ids: string[]): Promise<VectorDocument[]> {
    try {
      const url = `${this.collectionBaseUrl}/points`;
      const response = await requestUrl({
        url,
        method: "POST",
        contentType: "application/json",
        headers: {
          "api-key": `${this.qdrantApiKey}`,
        },
        body: JSON.stringify({
          ids,
          withPayload: true,
          withVectors: true,
        }),
        throw: false,
      });

      if (response.status < 200 || response.status >= 300) {
        const errorData =
          typeof response.json === "string" ? JSON.parse(response.json) : response.json;
        throw new Error(`Failed to retrieve points by IDs: ${JSON.stringify(errorData)}`);
      }

      const points = response.json.result;

      return points.map((point: any) => ({
        id: point.id.toString(),
        path: point.payload?.path as string,
        title: point.payload?.title as string,
        content: point.payload?.content as string,
        embedding: point.vector as number[],
        created_at: point.payload?.created_at as number,
        ctime: point.payload?.ctime as number,
        mtime: point.payload?.mtime as number,
        tags: point.payload?.tags as string[],
        extension: point.payload?.extension as string,
        nchars: point.payload?.nchars as number,
        embeddingModel: point.payload?.embeddingModel as string,
        metadata: point.payload as Record<string, any>,
      }));
    } catch (error) {
      logError("Error getting documents from Qdrant by IDs:", error);
      return [];
    }
  }

  public async getDocsJsonByPaths(paths: string[]): Promise<Record<string, any[]>> {
    try {
      const results: Record<string, any[]> = {};
      for (const path of paths) {
        const docs = await this.getDocsByPath(path);
        results[path] = docs;
      }
      return results;
    } catch (error) {
      logError("Error getting documents from Qdrant by paths:", error);
      return {};
    }
  }

  public async checkIndexIntegrity(): Promise<void> {
    // Implement Qdrant-specific index integrity checks here if needed

    if (!this.isInitialized) {
      throw new CustomError("Qdrant database not initialized.");
    }

    try {
      // Get all indexed files
      const indexedFiles = await this.getIndexedFiles();

      // Check each file for embeddings
      for (const filePath of indexedFiles) {
        const hasEmbeddings = await this.hasEmbeddings(filePath);
        if (!hasEmbeddings) {
          this.markFileMissingEmbeddings(filePath);
        }
      }

      const missingEmbeddings = this.getFilesMissingEmbeddings();
      if (missingEmbeddings.length > 0) {
        logInfo("Files missing embeddings after integrity check:", missingEmbeddings.join(", "));
      } else {
        logInfo("Index integrity check completed. All documents have embeddings.");
      }
    } catch (err) {
      logError("Error checking index integrity:", err);
      throw new CustomError("Failed to check index integrity.");
    }

    logInfo("Qdrant index integrity check completed.");
  }

  /**
   * Create a Payload Index for faster retrieval
   * @param fieldName
   */
  protected async createPayloadIndex(fieldName: string): Promise<void> {
    try {
      const url = `${this.collectionBaseUrl}/index`;
      const response = await requestUrl({
        url,
        method: "PUT",
        contentType: "application/json",
        headers: {
          "api-key": `${this.qdrantApiKey}`,
        },
        body: JSON.stringify({
          field_name: fieldName,
          field_schema: "keyword",
        }),
        throw: false,
      });

      if (response.status < 200 || response.status >= 300) {
        const errorData =
          typeof response.json === "string" ? JSON.parse(response.json) : response.json;
        throw new Error(
          `Failed to create payload index for field ${fieldName}: ${JSON.stringify(errorData)}`
        );
      }

      logInfo(
        `Created payload index for field: ${fieldName} in Qdrant collection: ${this.collectionName}`
      );
    } catch (error) {
      logError(`Error creating payload index for field ${fieldName} in Qdrant:`, error);
      throw error;
    }
  }

  private async ensureCollectionExists(): Promise<void> {
    if (!(await this.collectionExists())) {
      await this.createCollection();
    }
  }

  /**
   * Upsert a document to the database
   */
  public async upsert(
    doc: VectorDocument,
    commitToDB: boolean
  ): Promise<VectorDocument | undefined> {
    try {
      // Add to pending documents
      this.pendingDocuments.push(doc);

      // Update metadata cache
      this.updateMetadataCache(doc);

      // Save to DB if requested
      if (commitToDB) {
        await this.saveDB();
      } else if (this.pendingDocuments.length >= VECTORIZE_BATCH_LIMIT && this.isInitialized) {
        logInfo(`Reached batch limit of ${VECTORIZE_BATCH_LIMIT} documents, automatically saving`);
        await this.saveDB();
      }

      return doc;
    } catch (error) {
      logError(`Error adding document ${doc.id} to pending documents:`, error);
      return undefined;
    }
  }

  /**
   * Upsert multiple documents and save to the database
   */
  public async upsertManyAndSave(docs: VectorDocument[]): Promise<VectorDocument[]> {
    try {
      // Add to pending documents
      this.pendingDocuments.push(...docs);

      // Update metadata cache
      for (const doc of docs) {
        this.updateMetadataCache(doc);
      }

      // Always save when upsertManyAndSave is called
      if (this.isInitialized) {
        logInfo(`Reached batch limit of ${VECTORIZE_BATCH_LIMIT} documents, automatically saving`);
        await this.saveDB();
      }

      return docs;
    } catch (error) {
      logError(`Error adding multiple documents to pending documents:`, error);
      return [];
    }
  }
}
