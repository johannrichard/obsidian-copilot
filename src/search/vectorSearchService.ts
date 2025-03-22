import { App, Notice, TFile } from "obsidian";
import { DBProvider, VectorDocument, VectorSearchOptions } from "./dbProvider";
import EmbeddingsManager from "@/LLMProviders/embeddingManager";
import { logError, logInfo } from "@/logger";
import { Document } from "@langchain/core/documents";
import { extractNoteFiles } from "@/utils";
import DBOperationsManager from "./dbOperationsManager";

/**
 * Search service for vector databases
 * This service provides methods for searching across any vector database provider
 */
export class VectorSearchService {
  private static instance: VectorSearchService;
  private provider: DBProvider;
  private embeddingsManager: EmbeddingsManager;
  private isInitialized = false;

  private constructor(private app: App) {
    this.embeddingsManager = EmbeddingsManager.getInstance();
    this.provider = DBOperationsManager.getInstance(app).getProvider();
  }

  /**
   * Get the singleton instance of the search service
   */
  public static getInstance(app: App): VectorSearchService {
    if (!VectorSearchService.instance) {
      VectorSearchService.instance = new VectorSearchService(app);
    }
    return VectorSearchService.instance;
  }

  /**
   * Initialize the search service
   */
  public async initialize(): Promise<void> {
    if (this.isInitialized) {
      return;
    }

    try {
      const embeddingInstance = await this.embeddingsManager.getEmbeddingsAPI();
      if (!embeddingInstance) {
        throw new Error("Embedding instance not found.");
      }

      await this.provider.initializeDB(embeddingInstance);
      this.isInitialized = true;
      logInfo("Vector search service initialized successfully");
    } catch (error) {
      logError("Failed to initialize vector search service:", error);
      new Notice("Failed to initialize vector search service. Some features may be limited.");
    }
  }

  /**
   * Search for documents by query text
   * @param query The query text
   * @param options Search options
   * @returns Array of Document objects
   */
  public async searchByText(
    query: string,
    options: {
      limit?: number;
      similarity?: number;
      includeMetadata?: boolean;
      salientTerms?: string[];
    } = {}
  ): Promise<Document[]> {
    await this.ensureInitialized();

    const { limit = 5, similarity = 0.4, includeMetadata = true, salientTerms = [] } = options;

    try {
      // Extract note files from the query
      const noteFiles = extractNoteFiles(query, this.app.vault);

      // Get documents from explicitly mentioned notes
      const explicitDocs = await this.getDocumentsFromFiles(noteFiles);

      // If we have explicit documents and no salient terms, just return those
      if (explicitDocs.length > 0 && salientTerms.length === 0) {
        return explicitDocs;
      }

      // Convert query to embedding
      const embedding = await this.convertTextToEmbedding(query);

      // Search by embedding
      const searchOptions: VectorSearchOptions = {
        limit,
        similarity,
      };

      const results = await this.provider.getDocsByEmbedding(embedding, searchOptions);

      // Convert to Document objects
      const documents = this.convertToDocuments(results, includeMetadata);

      // Combine with explicit documents, ensuring no duplicates
      return this.combineAndDeduplicate(documents, explicitDocs);
    } catch (error) {
      logError("Error searching by text:", error);
      throw error;
    }
  }

  /**
   * Search for documents by path
   * @param path The file path
   * @returns Array of Document objects
   */
  public async searchByPath(
    path: string,
    options: { includeMetadata?: boolean } = {}
  ): Promise<Document[]> {
    await this.ensureInitialized();

    const { includeMetadata = true } = options;

    try {
      const results = await this.provider.getDocsByPath(path);
      return this.convertToDocuments(results, includeMetadata);
    } catch (error) {
      logError(`Error searching by path ${path}:`, error);
      throw error;
    }
  }

  /**
   * Get documents from files
   * @param files Array of TFile objects
   * @returns Array of Document objects
   */
  private async getDocumentsFromFiles(files: TFile[]): Promise<Document[]> {
    if (files.length === 0) {
      return [];
    }

    const documents: Document[] = [];

    for (const file of files) {
      try {
        const results = await this.provider.getDocsByPath(file.path);

        if (results.length > 0) {
          // Convert to Document objects and add to results
          const docs = this.convertToDocuments(results, true);
          documents.push(...docs);
        } else {
          // If no chunks found for this file, it might not be indexed
          logInfo(`No indexed chunks found for file: ${file.path}`);
        }
      } catch (error) {
        logError(`Error getting documents for file ${file.path}:`, error);
      }
    }

    return documents;
  }

  /**
   * Convert text to embedding vector
   * @param text The text to convert
   * @returns Embedding vector
   */
  private async convertTextToEmbedding(text: string): Promise<number[]> {
    try {
      const embeddingInstance = await this.embeddingsManager.getEmbeddingsAPI();
      if (!embeddingInstance) {
        throw new Error("Embedding instance not found.");
      }

      const embeddings = await embeddingInstance.embedQuery(text);
      return embeddings;
    } catch (error) {
      logError("Error converting text to embedding:", error);
      throw error;
    }
  }

  /**
   * Convert VectorDocument objects to Document objects
   * @param results Array of VectorDocument objects
   * @param includeMetadata Whether to include metadata in the Document objects
   * @returns Array of Document objects
   */
  private convertToDocuments(results: VectorDocument[], includeMetadata: boolean): Document[] {
    return results.map((doc) => {
      const metadata = includeMetadata
        ? {
            ...doc.metadata,
            path: doc.path,
            title: doc.title,
            id: doc.id,
            mtime: doc.mtime,
            ctime: doc.ctime,
            tags: doc.tags,
            extension: doc.extension,
            embeddingModel: doc.embeddingModel,
            created_at: doc.created_at,
            nchars: doc.nchars,
          }
        : { path: doc.path, title: doc.title };

      return new Document({
        pageContent: doc.content,
        metadata,
      });
    });
  }

  /**
   * Combine and deduplicate Document arrays
   * @param documents1 First array of Document objects
   * @param documents2 Second array of Document objects
   * @returns Combined and deduplicated array of Document objects
   */
  private combineAndDeduplicate(documents1: Document[], documents2: Document[]): Document[] {
    const combined = [...documents1, ...documents2];

    // Use a Map to deduplicate by id
    const uniqueMap = new Map<string, Document>();

    for (const doc of combined) {
      const id = doc.metadata.id;
      if (id && !uniqueMap.has(id)) {
        uniqueMap.set(id, doc);
      }
    }

    return Array.from(uniqueMap.values());
  }

  /**
   * Ensure the search service is initialized
   */
  private async ensureInitialized(): Promise<void> {
    if (!this.isInitialized) {
      await this.initialize();
    }
  }

  /**
   * Get the current provider
   */
  public getProvider(): DBProvider {
    return this.provider;
  }

  /**
   * Update the provider
   * @param providerType The provider type
   */
  public async updateProvider(providerType: string): Promise<void> {
    this.provider = DBOperationsManager.getInstance(app).getProvider();
    this.isInitialized = false;
    await this.initialize();
  }
}
