import { VECTORIZE_BATCH_LIMIT } from "@/constants";
import { App, Notice, Platform, requestUrl } from "obsidian";
import { VectorDocument, VectorDocumentMetadata } from "./dbProvider";
import { Embeddings } from "@langchain/core/embeddings";
import { logError, logInfo } from "@/logger";
import { CustomError } from "@/error";
import EmbeddingsManager from "@/LLMProviders/embeddingManager";
import { getVectorLength } from "./searchUtils";
import { getSettings, subscribeToSettingsChange } from "@/settings/model";
import { BaseCloudDBProvider } from "./BaseCloudDBProvider";

/**
 * Cloudflare Vectorize implementation of DBProvider
 * This provider uses Cloudflare's Vectorize API for vector storage and search
 */
export class CloudflareDBProvider extends BaseCloudDBProvider {
  protected isInitialized = false;
  protected isInitializing = false;
  protected initializationError: Error | null = null;
  protected filesWithoutEmbeddings: Set<string> = new Set();
  protected lastInitAttempt = 0;
  protected embeddingInstance: Embeddings;
  protected vectorLength: number = 0;
  protected embeddingModelName: string = "";
  protected namespace: string = "obsidian-copilot";
  protected pendingDocuments: VectorDocument[] = []; // In-memory store for pending documents
  protected metadataCache: Record<string, VectorDocumentMetadata> = {}; // Cache of document metadata
  protected metadataPath: string = ""; // Path to metadata file
  protected metadataLastSaved: number = 0; // Timestamp of last metadata save
  private cloudflareApiToken: string = "";
  private cloudflareAccountId: string = "";
  private cloudflareVectorizeUrl: string;
  private indexUrl: string = "";

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
        await this.initializeDB(await EmbeddingsManager.getInstance().getEmbeddingsAPI());
      }
    });
  }

  protected initializationPromise: Promise<void>;
  protected getDocsFromCloudById(ids: string[]): Promise<VectorDocument[]> {
    throw new Error("Method not implemented.");
  }
  public waitForInitialization(): Promise<void> {
    throw new Error("Method not implemented.");
  }
  public checkAndHandleEmbeddingModelChange(embeddingInstance: Embeddings): Promise<boolean> {
    throw new Error("Method not implemented.");
  }
  public getDocsByPath(path: string): Promise<VectorDocument[]> {
    throw new Error("Method not implemented.");
  }
  getIndexedFiles(): Promise<string[]> {
    try {
      // Search all documents and get unique file path;
      const docs = Object.values(this.metadataCache);

      // Use a Set to get unique file paths since multiple chunks can belong to the same file
      const uniquePaths = new Set<string>();
      docs.forEach((doc) => {
        uniquePaths.add(doc.path);
      });

      // Convert Set to sorted array
      return Promise.resolve(Array.from(uniquePaths).sort());
    } catch (err) {
      logError("Error getting indexed files:", err);
      throw new CustomError("Failed to retrieve indexed files.");
    }
  }

  hasIndex(notePath: string): Promise<boolean> {
    // Check Metadata cache

    const docs = Object.values(this.metadataCache);
    const doc = docs.filter((doc) => {
      return doc.path === notePath;
    });

    return Promise.resolve(doc.length > 0);
  }

  getDocsJsonByPaths(paths: string[]): Promise<Record<string, any[]>> {
    // Metadata match and then fetch the documents by ID
    throw new Error("Method not implemented.");
  }

  checkIndexIntegrity(): Promise<void> {
    throw new Error("Method not implemented.");
  }

  async initializeDB(embeddingInstance: Embeddings): Promise<void> {
    // Store the embedding instance for later retry if needed
    this.embeddingInstance = embeddingInstance;

    // Don't attempt to initialize if already initializing
    if (this.isInitializing) {
      return;
    }

    // Reset initialization state
    this.isInitializing = true;
    this.initializationError = null;
    this.lastInitAttempt = Date.now();

    try {
      // Get Cloudflare credentials from settings
      const settings = getSettings();
      this.cloudflareApiToken = settings.cloudflareApiToken || "";
      this.cloudflareAccountId = settings.cloudflareAccountId || "";

      this.namespace = "obsidian-copilot"; // Fixed namespace for now, could be made configurable in the future

      if (!this.cloudflareApiToken || !this.cloudflareAccountId) {
        throw new CustomError(
          "Cloudflare API token or Account ID not configured. Please set them in the settings."
        );
      }

      // Set up the Vectorize API URL with V2 endpoint
      this.cloudflareVectorizeUrl = `https://api.cloudflare.com/client/v4/accounts/${this.cloudflareAccountId}/vectorize/v2/indexes`;
      this.indexUrl = `${this.cloudflareVectorizeUrl}/${this.collectionName}`;

      // Load metadata if it exists - this doesn't require network connectivity
      await this.loadMetadata();

      // Get embedding model information - this requires network connectivity
      await this.initializeEmbeddingModel(embeddingInstance);

      // Check if the index exists, create it if it doesn't - this requires network connectivity
      await this.initializeCloudflareIndex();

      this.isInitialized = true;
      this.isInitializing = false;
      logInfo(
        `Initialized Cloudflare Vectorize V2 provider with model: ${this.embeddingModelName}`
      );
    } catch (error) {
      this.isInitializing = false;
      this.isInitialized = false;
      this.initializationError = error;
      logError("Failed to initialize Cloudflare Vectorize provider:", error);
      new Notice(
        "Failed to initialize Cloudflare Vectorize provider. Will retry when network is available."
      );
      // Don't throw here to allow for deferred initialization
    }
  }

  /**
   * Initialize the embedding model information
   */
  private async initializeEmbeddingModel(embeddingInstance: Embeddings): Promise<void> {
    try {
      this.embeddingModelName = EmbeddingsManager.getModelName(embeddingInstance);
      this.vectorLength = await getVectorLength(embeddingInstance);

      if (!this.vectorLength || this.vectorLength === 0) {
        throw new CustomError(
          "Invalid vector length detected. Please check if your embedding model is working."
        );
      }

      logInfo(
        `Successfully initialized embedding model: ${this.embeddingModelName} with vector length: ${this.vectorLength}`
      );
    } catch (error) {
      logError("Failed to initialize embedding model:", error);
      throw new CustomError(`Failed to initialize embedding model: ${error.message}`);
    }
  }

  /**
   * Initialize the Cloudflare Vectorize index
   */
  private async initializeCloudflareIndex(): Promise<void> {
    try {
      // Check if the index exists, create it if it doesn't
      if (!(await this.collectionExists())) {
        await this.createCollection();
      } else {
        // Validate the index dimensions
        const isValid = await this.validateVectorDimensions();
        if (!isValid) {
          if (this.embeddingInstance) {
            await this.recreateIndex(this.embeddingInstance);
          } else {
            throw new CustomError("Cannot recreate index: embedding instance is null");
          }
        }
      }
      logInfo(`Successfully connected to Cloudflare Vectorize index: ${this.collectionName}`);
    } catch (error) {
      logError("Failed to initialize Cloudflare Vectorize index:", error);
      throw new CustomError(`Failed to initialize Cloudflare Vectorize index: ${error.message}`);
    }
  }

  /**
   * Manually trigger a retry of initialization
   * This can be called when network connectivity is restored
   */
  async retryInitialization(): Promise<boolean> {
    if (this.isInitialized || this.isInitializing || !this.embeddingInstance) {
      return this.isInitialized;
    }

    try {
      await this.initializeDB(this.embeddingInstance);
      return this.isInitialized;
    } catch (error) {
      logError("Failed to initialize during manual retry:", error);
      return false;
    }
  }

  async onunload(): Promise<void> {
    // Save any pending documents before shutdown
    if (this.isInitialized && this.pendingDocuments.length > 0) {
      try {
        logInfo(`Saving ${this.pendingDocuments.length} pending documents before shutdown`);
        await this.saveDB();
      } catch (error) {
        logError("Error saving pending documents during shutdown:", error);
      }
    }

    // Save metadata before shutdown
    if (this.isInitialized && Object.keys(this.metadataCache).length > 0) {
      try {
        await this.saveMetadata();
      } catch (error) {
        logError("Error saving metadata during shutdown:", error);
      }
    }

    this.isInitialized = false;
  }

  async saveDB(): Promise<void> {
    // Try to ensure we're initialized before saving
    const initialized = await this.ensureInitialized();

    if (!initialized) {
      logInfo(
        "Cloudflare Vectorize provider not initialized, storing documents in memory until connectivity is available"
      );
      return;
    }

    if (this.pendingDocuments.length === 0) {
      logInfo("No pending documents to save to Cloudflare Vectorize");
      return;
    }

    try {
      // Process documents in batches to respect Cloudflare API limits
      const totalDocuments = this.pendingDocuments.length;
      logInfo(`Saving ${totalDocuments} pending documents to Cloudflare Vectorize`);

      // Update metadata cache with pending documents
      for (const doc of this.pendingDocuments) {
        this.updateMetadataCache(doc);
      }

      // Process in batches of BATCH_LIMIT
      for (let i = 0; i < totalDocuments; i += VECTORIZE_BATCH_LIMIT) {
        const batch = this.pendingDocuments.slice(i, i + VECTORIZE_BATCH_LIMIT);
        await this.upsertToCloud(batch);
        logInfo(
          `Saved batch of ${batch.length} documents to Cloudflare Vectorize (${i + batch.length}/${totalDocuments})`
        );
      }

      // Clear the pending documents after successful save
      this.pendingDocuments = [];

      // Save metadata after successful save
      await this.saveMetadata();

      logInfo("All pending documents successfully saved to Cloudflare Vectorize");
    } catch (error) {
      logError("Error saving pending documents to Cloudflare Vectorize:", error);
      new Notice("An error occurred while saving to Cloudflare Vectorize. Will retry later.");
      // Don't throw here to allow for deferred saving
    }
  }

  async recreateIndex(embeddingInstance: Embeddings): Promise<void> {
    try {
      // Store the embedding instance for later use
      this.embeddingInstance = embeddingInstance;

      // Clear metadata cache
      this.reinitializeMetadataCache();

      // Clear the in-memory store
      const pendingCount = this.pendingDocuments.length;
      if (pendingCount > 0) {
        logInfo(`Clearing ${pendingCount} pending documents from memory`);
        this.pendingDocuments = [];
      }

      // If we're not initialized, we can't delete the remote index
      if (!this.isInitialized) {
        await this.initializeDB(embeddingInstance);
        logInfo("Cloudflare Vectorize provider not initialized, skipping remote index deletion");
        new Notice(
          "Local data cleared. Remote index will be cleared when connectivity is restored."
        );
        return;
      }

      // Delete the index and recreate it
      await this.clearCollection();
      await this.initializeDB(embeddingInstance);

      new Notice("Cloudflare Vectorize index cleared successfully.");
      logInfo("Cloudflare Vectorize index cleared successfully.");
    } catch (error) {
      logError("Error clearing Cloudflare Vectorize index:", error);
      new Notice("An error occurred while clearing the Cloudflare Vectorize index.");
      throw error;
    }
  }

  async getStoragePath(): Promise<string> {
    return `Cloudflare Vectorize (${this.collectionName})`;
  }

  async upsert(doc: VectorDocument, commitToDB: boolean): Promise<VectorDocument | undefined> {
    // Try to ensure we're initialized, but don't block if not
    await this.ensureInitialized();

    try {
      // Add to pending documents
      this.pendingDocuments.push(doc);

      // Update metadata cache
      this.updateMetadataCache(doc);

      logInfo(
        `Added document ${doc.id} to pending documents (total: ${this.pendingDocuments.length})`
      );

      // If we're asked to commit or if we've reached the batch limit and we're initialized, automatically save
      if (
        (commitToDB || this.pendingDocuments.length >= VECTORIZE_BATCH_LIMIT) &&
        this.isInitialized
      ) {
        logInfo(`Reached batch limit of ${VECTORIZE_BATCH_LIMIT} documents, automatically saving`);
        await this.saveDB();
      }

      return doc;
    } catch (error) {
      logError(`Error adding document ${doc.id} to pending documents:`, error);
      return undefined;
    }
  }

  async upsertManyAndSave(docs: VectorDocument[]): Promise<VectorDocument[]> {
    // Try to ensure we're initialized, but don't block if not
    await this.ensureInitialized();

    if (!docs || docs.length === 0) {
      return [];
    }

    try {
      logInfo(`Adding ${docs.length} documents to pending documents`);

      // Add all documents to pending
      this.pendingDocuments.push(...docs);

      // Update metadata cache for all documents
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

  async removeDocs(path: string): Promise<void> {
    // Try to ensure we're initialized, but don't block if not
    await this.ensureInitialized();

    try {
      // Remove matching documents from the pending documents
      const initialPendingCount = this.pendingDocuments.length;
      this.pendingDocuments = this.pendingDocuments.filter((doc) => doc.path !== path);
      const removedFromPending = initialPendingCount - this.pendingDocuments.length;

      if (removedFromPending > 0) {
        logInfo(
          `Removed ${removedFromPending} documents with path: ${path} from pending documents`
        );
      }

      // Remove from metadata cache
      if (path in this.metadataCache) {
        delete this.metadataCache[path];
        logInfo(`Removed path ${path} from metadata cache`);
      }

      // If not initialized, we can't remove from Cloudflare
      if (!this.isInitialized) {
        logInfo("Cloudflare Vectorize provider not initialized, skipping remote removal");
        return;
      }

      // Find all documents with the given path in the database
      const docsToRemove: VectorDocument[] = await this.getDocsByPath(path);

      if (docsToRemove.length === 0 && removedFromPending === 0) {
        return;
      }

      // Remove from Cloudflare Vectorize V2
      for (const doc of docsToRemove) {
        await this.deleteFromCloud(doc.id);
      }

      if (docsToRemove.length > 0) {
        logInfo(
          `Removed ${docsToRemove.length} documents with path: ${path} from Cloudflare Vectorize V2`
        );

        // Save metadata after removal
        await this.saveMetadata();
      }
    } catch (error) {
      logError(`Error removing documents with path ${path} from Cloudflare Vectorize:`, error);
    }
  }

  public getDb(): void | undefined {
    if (!this.isInitialized) {
      console.warn("Database not initialized. Some features may be limited.");
    }

    throw new Error("Method not implemented.");
  }

  public getDbPath(): Promise<string> {
    return Promise.resolve(this.indexUrl);
  }

  public async getIsIndexLoaded(): Promise<boolean> {
    // We don't "load" the index
    return this.isInitialized;
  }

  public getCurrentDbPath(): string {
    // This is the old path before any setting changes, used for comparison
    // In principle it should never change unless the vault is renamed
    return this.indexUrl;
  }

  // Helper method to calculate cosine similarity between two vectors
  protected calculateCosineSimilarity(a: number[], b: number[]): number {
    if (!a || !b || a.length !== b.length) {
      return 0;
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    normA = Math.sqrt(normA);
    normB = Math.sqrt(normB);

    if (normA === 0 || normB === 0) {
      return 0;
    }

    return dotProduct / (normA * normB);
  }

  public async getLatestFileMtime(): Promise<number> {
    if (!this.isInitialized) {
      throw new CustomError("Cloudflare Vectorize provider not initialized");
    }

    try {
      let latestMtime = 0;

      // Check metadata cache for the latest mtime
      for (const path in this.metadataCache) {
        const mtime = this.metadataCache[path].mtime || 0;
        if (mtime > latestMtime) {
          latestMtime = mtime;
        }
      }

      // Also check pending documents that might not be in the metadata cache yet
      for (const doc of this.pendingDocuments) {
        const mtime = doc.mtime || 0;
        if (mtime > latestMtime) {
          latestMtime = mtime;
        }
      }

      return latestMtime;
    } catch (error) {
      logError("Error getting latest mtime from metadata cache:", error);
      return 0;
    }
  }

  public async getAllDocuments(): Promise<VectorDocument[]> {
    if (!this.isInitialized) {
      throw new CustomError("Cloudflare Vectorize provider not initialized");
    }

    try {
      // Get all documents from metadata cache
      const documents: VectorDocument[] = [];

      // First add documents from metadata cache
      for (const [id, metadata] of Object.entries(this.metadataCache)) {
        // Create a lightweight document with essential information
        documents.push({
          id: id,
          path: metadata.path,
          embeddingModel: metadata.embeddingModel,
          mtime: metadata.mtime,
          // Add minimal required fields with default values
          title: metadata.path.split("/").pop() || "",
          content: "",
          embedding: [],
          created_at: metadata.mtime,
          ctime: metadata.mtime,
          tags: [],
          extension: metadata.path.split(".").pop() || "",
          nchars: 0,
          metadata: {},
        });
      }

      // Then add any pending documents that might not be in the metadata yet
      for (const doc of this.pendingDocuments) {
        // Check if this document is already included from metadata
        const existingIndex = documents.findIndex((d) => d.id === doc.id);

        if (existingIndex >= 0) {
          // Update the existing document with more complete information
          documents[existingIndex] = doc;
        } else {
          // Add the pending document
          documents.push(doc);
        }
      }

      logInfo(`Retrieved ${documents.length} documents from metadata cache and pending documents`);
      return documents;
    } catch (error) {
      logError("Error getting all documents from metadata cache:", error);
      return this.pendingDocuments; // Return at least the pending documents if metadata retrieval fails
    }
  }

  async hasDocument(path: string): Promise<boolean> {
    // Try to ensure we're initialized, but don't block if not
    await this.ensureInitialized();

    if (!path) return false;

    try {
      // First check the pending documents
      const hasPendingDoc = this.pendingDocuments.some((doc) => doc.path === path);
      if (hasPendingDoc) {
        return true;
      }

      // Then check metadata cache
      if (path in this.metadataCache) {
        return true;
      }

      // If not initialized, we can only check local data
      if (!this.isInitialized) {
        return false;
      }

      // Then search for documents with the given path in the database
      const docs: VectorDocument[] = await this.getDocsByPath(path);
      return docs.length > 0;
    } catch (error) {
      logError(`Error checking if document exists in Cloudflare Vectorize: ${path}`, error);
      return false;
    }
  }

  async hasEmbeddings(path: string): Promise<boolean> {
    // Try to ensure we're initialized, but don't block if not
    await this.ensureInitialized();

    if (!path) return false;

    // First check the pending documents
    const pendingDocs = this.pendingDocuments.filter((doc) => doc.path === path);
    if (pendingDocs.length > 0) {
      return pendingDocs.every(
        (doc) => doc.embedding && Array.isArray(doc.embedding) && doc.embedding.length > 0
      );
    }

    // Then check if the path exists in the metadata cache
    if (path in this.metadataCache) {
      // If not initialized, we can't check the actual embeddings
      if (!this.isInitialized) {
        // Assume it has embeddings if it's in the metadata cache
        return true;
      }

      const docId = this.metadataCache[path].id;

      try {
        // Search for document by ID using the API
        const results = await this.getDocsById([docId]);

        if (!results || !Array.isArray(results) || results.length === 0) {
          return false;
        }

        // Check if all vectors have values
        return results.every(
          (doc: any) => doc.values && Array.isArray(doc.values) && doc.values.length > 0
        );
      } catch (error) {
        logError(`Error checking embeddings for path ${path}:`, error);
        return false;
      }
    }

    return false;
  }

  private async getDocsById(docIds: string[]): Promise<VectorDocument[]> {
    try {
      const results: VectorDocument[] = [];
      const response = await requestUrl({
        url: `${this.indexUrl}/vectors/get_by_ids`,
        method: "POST",
        contentType: "application/json",
        headers: {
          Authorization: `Bearer ${this.cloudflareApiToken}`,
        },
        body: JSON.stringify({
          namespace: this.namespace,
          ids: docIds,
        }),
        throw: false,
      });

      if (response.status < 200 || response.status >= 300) {
        return [];
      }

      const data = typeof response.json === "string" ? JSON.parse(response.json) : response.json;

      if (data.result && data.success) {
        data.result.map((doc: any) => {
          const metadata = doc.metadata || {};
          const vd: VectorDocument = {
            id: doc.id,
            title: metadata.title || "",
            content: metadata.content || "",
            embedding: doc.values || [],
            path: metadata.path || "",
            embeddingModel: metadata.embeddingModel || "",
            created_at: metadata.created_at,
            ctime: metadata.ctime,
            mtime: metadata.mtime,
            tags: metadata.tags || [],
            extension: metadata.extension || "",
            nchars: metadata.nchars || 0,
            metadata: metadata,
          };

          results.push(vd);
        });
      }

      return results;
    } catch (error) {
      logError(`Error getting documents by ID from Cloudflare Vectorize:`, error);
      return [];
    }
  }

  protected async createCollection(): Promise<void> {
    try {
      const response = await requestUrl({
        url: this.cloudflareVectorizeUrl,
        method: "POST",
        contentType: "application/json",
        headers: {
          Authorization: `Bearer ${this.cloudflareApiToken}`,
        },
        body: JSON.stringify({
          config: {
            dimensions: this.vectorLength,
            metric: "cosine",
          },
          name: this.collectionName,
          description: "Obsidian Copilot vector index for embeddings",
        }),
        throw: false,
      });

      if (response.status < 200 || response.status >= 300) {
        const errorData =
          typeof response.json === "string" ? JSON.parse(response.json) : response.json;
        throw new Error(`Failed to create index: ${JSON.stringify(errorData)}`);
      }

      logInfo(`Created Cloudflare Vectorize index: ${this.collectionName}`);
    } catch (error) {
      logError("Error creating Cloudflare Vectorize index:", error);
      throw error;
    }
  }

  protected async validateVectorDimensions(): Promise<boolean> {
    try {
      const response = await requestUrl({
        url: `${this.cloudflareVectorizeUrl}`,
        method: "GET",
        headers: {
          Authorization: `Bearer ${this.cloudflareApiToken}`,
        },
        throw: false,
      });

      if (response.status < 200 || response.status >= 300) {
        const errorData =
          typeof response.json === "string" ? JSON.parse(response.json) : response.json;
        throw new Error(`Failed to list indexes: ${JSON.stringify(errorData)}`);
      }

      const data = typeof response.json === "string" ? JSON.parse(response.json) : response.json;

      if (!data.result || !Array.isArray(data.result)) {
        throw new Error(`Invalid response format: ${JSON.stringify(data)}`);
      }

      const indexInfo = data.result.find((index: any) => index.name === this.collectionName);

      if (!indexInfo) {
        logInfo(`Index ${this.collectionName} not found.`);
        return false;
      }

      const dimensions = indexInfo.config.dimensions;

      if (dimensions !== this.vectorLength) {
        logInfo(
          `Index ${this.collectionName} has incorrect dimensions: ${dimensions}, expected: ${this.vectorLength}`
        );
        return false;
      }

      return true;
    } catch (error) {
      logError("Error validating index dimensions:", error);
      throw error;
    }
  }

  protected async upsertToCloud(docs: VectorDocument[]): Promise<void> {
    try {
      if (docs.length === 0) {
        return;
      }

      // Create NDJSON by converting each document to JSON and joining with newlines
      const ndjsonBody = docs
        .map((doc) => {
          // Prepare metadata and ensure arrays only contain strings (Vectorize limitation)
          const processedMetadata = { ...doc.metadata };
          for (const [key, value] of Object.entries(processedMetadata)) {
            if (Array.isArray(value)) {
              processedMetadata[key] = value.map((item) => String(item));
            }
          }

          const metadata = {
            title: doc.title,
            path: doc.path,
            content: doc.content,
            embeddingModel: doc.embeddingModel,
            created_at: doc.created_at,
            ctime: doc.ctime,
            mtime: doc.mtime,
            tags: doc.tags.map((tag) => String(tag)), // Ensure tags are strings
            extension: doc.extension,
            nchars: doc.nchars,
            ...processedMetadata,
          };

          // Create the vector object
          return JSON.stringify({
            id: doc.id,
            values: doc.embedding,
            metadata: metadata,
            namespace: this.namespace,
          });
        })
        .join("\n");

      // Upsert vectors to Cloudflare Vectorize
      const response = await requestUrl({
        url: `${this.indexUrl}/upsert`,
        method: "POST",
        headers: {
          Authorization: `Bearer ${this.cloudflareApiToken}`,
          "Content-Type": "application/x-ndjson",
        },
        body: ndjsonBody,
        throw: false,
      });

      if (response.status < 200 || response.status >= 300) {
        const errorData =
          typeof response.json === "string" ? JSON.parse(response.json) : response.json;
        throw new Error(`Failed to upsert vectors: ${JSON.stringify(errorData)}`);
      }

      logInfo(`Successfully upserted ${docs.length} vectors to Cloudflare Vectorize`);
    } catch (error) {
      logError(`Error upserting vectors to Cloudflare Vectorize`, error);
      throw error;
    }
  }

  protected async deleteFromCloud(id: string): Promise<void> {
    try {
      const response = await requestUrl({
        url: `${this.indexUrl}/vectors/delete`,
        method: "POST",
        contentType: "application/json",
        headers: {
          Authorization: `Bearer ${this.cloudflareApiToken}`,
        },
        body: JSON.stringify({
          ids: [id],
          namespace: this.namespace,
        }),
        throw: false,
      });

      if (response.status < 200 || response.status >= 300) {
        const errorData =
          typeof response.json === "string" ? JSON.parse(response.json) : response.json;
        throw new Error(`Failed to delete vector: ${JSON.stringify(errorData)}`);
      }
    } catch (error) {
      logError(`Error deleting vector from Cloudflare Vectorize: ${id}`, error);
      throw error;
    }
  }

  protected async clearCollection(): Promise<void> {
    try {
      logInfo(`Deleting Cloudflare Vectorize V2 index: ${this.collectionName}`);

      // Delete the entire index
      const response = await requestUrl({
        url: this.indexUrl,
        method: "DELETE",
        headers: {
          Authorization: `Bearer ${this.cloudflareApiToken}`,
        },
        throw: false,
      });

      if (response.status < 200 || (response.status >= 300 && response.status !== 404)) {
        // If it's a 404, the index doesn't exist anyway, so we can proceed
        const errorData =
          typeof response.json === "string" ? JSON.parse(response.json) : response.json;
        throw new Error(`Failed to delete index: ${JSON.stringify(errorData)}`);
      }

      logInfo(`Deleted Cloudflare Vectorize index: ${this.collectionName}`);
    } catch (error) {
      logError("Error deleting Cloudflare Vectorize index:", error);
      throw error;
    }
  }

  protected async searchCloudByEmbedding(
    embedding: number[],
    limit: number,
    threshold: number
  ): Promise<VectorDocument[]> {
    try {
      logInfo("Searching Cloudflare Vectorize");

      const response = await requestUrl({
        url: `${this.indexUrl}/query`,
        method: "POST",
        contentType: "application/json",
        headers: {
          Authorization: `Bearer ${this.cloudflareApiToken}`,
        },
        body: JSON.stringify({
          vector: embedding,
          topK: limit,
          returnValues: true,
          returnMetadata: "all",
          namespace: this.namespace,
        }),
        throw: false,
      });

      if (response.status < 200 || response.status >= 300) {
        const errorData =
          typeof response.json === "string" ? JSON.parse(response.json) : response.json;
        throw new Error(`Failed to query vectors: ${JSON.stringify(errorData)}`);
      }

      const data = typeof response.json === "string" ? JSON.parse(response.json) : response.json;

      if (!data.result || !Array.isArray(data.result.matches)) {
        return [];
      }

      // Filter by similarity threshold and convert to VectorDocument
      return data.result.matches
        .filter((match: { score: number }) => match.score >= threshold)
        .map(
          (match: {
            id: string;
            values: number[];
            metadata: Record<string, any>;
            score: number;
          }) => {
            const metadata = match.metadata || {};
            return {
              id: match.id,
              title: metadata.title || "",
              content: metadata.content || "",
              embedding: match.values || [],
              path: metadata.path || "",
              embeddingModel: metadata.embeddingModel || this.embeddingModelName,
              created_at: metadata.created_at || Date.now(),
              ctime: metadata.ctime || Date.now(),
              mtime: metadata.mtime || Date.now(),
              tags: metadata.tags || [],
              extension: metadata.extension || "",
              nchars: metadata.nchars || 0,
              metadata: metadata,
            };
          }
        );
    } catch (error) {
      logError("Error searching vectors in Cloudflare Vectorize:", error);
      throw error;
    }
  }

  protected async searchCloudByPath(path: string): Promise<VectorDocument[]> {
    throw new Error(`Method not implemented. Won't search for ${path} in Cloudflare Vectorize.`);
  }
  /**
   * Checks if a file is indexed based on metadata
   */
  async isFileIndexed(path: string): Promise<boolean> {
    if (!this.isInitialized) {
      return false;
    }

    // Check pending documents first
    const isPending = this.pendingDocuments.some((doc) => doc.path === path);
    if (isPending) {
      return true;
    }

    // Then check metadata cache
    return path in this.metadataCache;
  }

  /**
   * Gets all indexed file paths from metadata
   */
  async getAllIndexedFilePaths(): Promise<string[]> {
    if (!this.isInitialized) {
      return [];
    }

    // Get paths from metadata cache
    const metadataPaths = Object.keys(this.metadataCache);

    // Get paths from pending documents that aren't in metadata yet
    const pendingPaths = this.pendingDocuments
      .map((doc) => doc.path)
      .filter((path) => !metadataPaths.includes(path));

    return [...metadataPaths, ...pendingPaths];
  }

  /**
   * Performs garbage collection by removing documents that no longer exist
   */
  async garbageCollect(): Promise<number> {
    if (!this.isInitialized) {
      return 0;
    }

    try {
      // Find paths in metadata that don't exist anymore
      const pathsToRemove = Object.keys(this.metadataCache).filter(
        (path) => this.app.vault.getAbstractFileByPath(path) === null
      );

      if (pathsToRemove.length === 0) {
        logInfo("No documents to garbage collect");
        return 0;
      }

      logInfo(`Found ${pathsToRemove.length} documents to garbage collect`);

      // Remove each document
      for (const path of pathsToRemove) {
        await this.removeDocs(path);
      }

      return pathsToRemove.length;
    } catch (error) {
      logError("Error during garbage collection:", error);
      return 0;
    }
  }

  /**
   * Checks if the index exists
   */
  protected async collectionExists(): Promise<boolean> {
    // Check if we have the necessary configuration
    if (!this.cloudflareApiToken || !this.cloudflareAccountId || !this.indexUrl) {
      return false;
    }

    try {
      const response = await requestUrl({
        url: this.indexUrl,
        method: "GET",
        headers: {
          Authorization: `Bearer ${this.cloudflareApiToken}`,
        },
        throw: false,
      });

      return response.status >= 200 && response.status < 300;
    } catch {
      return false;
    }
  }
}
