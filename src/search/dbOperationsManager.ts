import { App, Notice, TFile } from "obsidian";
import { DBProvider as DBProvider, VectorDocument, VectorSearchOptions } from "./dbProvider";
import { OramaDBProvider } from "./oramaDBProvider";
import { CloudflareDBProvider as CloudflareDBProvider } from "./cloudflareDBProvider";
import { QdrantDBProvider } from "./qdrantDBPRovider";
import { getSettings, subscribeToSettingsChange } from "@/settings/model";
import EmbeddingsManager from "@/LLMProviders/embeddingManager";
import { logError, logInfo } from "@/logger";
import { Document } from "@langchain/core/documents";
import { extractNoteFiles } from "@/utils";
import { CustomError } from "@/error";

// Define the vector database provider types
export enum DBProviderTypes {
  ORAMA = "orama",
  CLOUDFLARE = "cloudflare",
  QDRANT = "qdrant",
}

// Define the provider constructor map
type DBProviderConstructorType = new (app: App) => DBProvider;

const DB_PROVIDER_CONSTRUCTORS: Record<DBProviderTypes, DBProviderConstructorType> = {
  [DBProviderTypes.ORAMA]: OramaDBProvider,
  [DBProviderTypes.CLOUDFLARE]: CloudflareDBProvider,
  [DBProviderTypes.QDRANT]: QdrantDBProvider,
};

/**
 * Manager for vector database operations
 * This class combines the functionality of vectorDBFactory and vectorSearchService
 */
export default class DBOperationsManager {
  private static instance: DBOperationsManager;
  private provider: DBProvider;
  private embeddingsManager: EmbeddingsManager;
  private isInitialized = false;
  private static providerMap: Record<
    string,
    {
      ProviderConstructor: DBProviderConstructorType;
    }
  > = {};

  private constructor(private app: App) {
    this.embeddingsManager = EmbeddingsManager.getInstance();
    this.initialize();
    subscribeToSettingsChange(() => this.updateProvider());
  }

  /**
   * Get the singleton instance of the vector database manager
   */
  public static getInstance(app: App): DBOperationsManager {
    if (!DBOperationsManager.instance) {
      DBOperationsManager.instance = new DBOperationsManager(app);
    }
    return DBOperationsManager.instance;
  }

  /**
   * Initialize the vector database manager
   */
  private async initialize(): Promise<void> {
    this.buildProviderMap();
    await this.updateProvider();
  }

  /**
   * Build the provider map
   */
  private buildProviderMap(): void {
    DBOperationsManager.providerMap = {};

    // Add all available providers to the map
    Object.values(DBProviderTypes).forEach((providerType) => {
      const constructor = DB_PROVIDER_CONSTRUCTORS[providerType as DBProviderTypes];
      if (constructor) {
        DBOperationsManager.providerMap[providerType] = {
          ProviderConstructor: constructor,
        };
      }
    });
  }

  /**
   * Update the current provider based on settings
   */
  public async updateProvider(): Promise<void> {
    const providerType = getSettings().vectorDbType || DBProviderTypes.ORAMA;

    if (!DBOperationsManager.providerMap[providerType]) {
      throw new CustomError(`Unknown vector database provider type: ${providerType}`);
    }

    // Create the new provider
    const { ProviderConstructor } = DBOperationsManager.providerMap[providerType];
    this.provider = new ProviderConstructor(this.app);

    // Reset initialization flag
    this.isInitialized = false;

    // Initialize the provider if we have embeddings
    try {
      const embeddingInstance = await this.embeddingsManager.getEmbeddingsAPI();
      if (embeddingInstance) {
        await this.provider.initializeDB(embeddingInstance);
        this.isInitialized = true;
        logInfo(`Vector database provider (${providerType}) initialized successfully`);
      }
    } catch (error) {
      logError(`Failed to initialize vector database provider (${providerType}):`, error);
      new Notice(`Failed to initialize vector database provider. Some features may be limited.`);
    }
  }

  /**
   * Get the current provider
   */
  public getProvider(): DBProvider {
    return this.provider;
  }

  /**
   * Create a provider of the specified type
   */
  public createProvider(providerType: string): DBProvider {
    if (!DBOperationsManager.providerMap[providerType]) {
      throw new CustomError(`Unknown vector database provider type: ${providerType}`);
    }

    const { ProviderConstructor } = DBOperationsManager.providerMap[providerType];
    return new ProviderConstructor(this.app);
  }

  /**
   * Ensure the manager is initialized
   */
  private async ensureInitialized(): Promise<void> {
    if (!this.isInitialized) {
      await this.initialize();

      if (!this.isInitialized) {
        throw new CustomError("Vector database manager is not initialized");
      }
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
      const vectorDocs = this.convertToDocuments(results, includeMetadata);

      // Combine and deduplicate results
      return this.combineAndDeduplicate(explicitDocs, vectorDocs);
    } catch (error) {
      logError("Error searching by text:", error);
      throw new CustomError(`Failed to search by text: ${error.message}`);
    }
  }

  /**
   * Search for documents by path
   * @param path The file path
   * @param options Search options
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
      logError("Error searching by path:", error);
      throw new CustomError(`Failed to search by path: ${error.message}`);
    }
  }

  /**
   * Get documents from files
   * @param files Array of TFile objects
   * @returns Array of Document objects
   */
  private async getDocumentsFromFiles(files: TFile[]): Promise<Document[]> {
    if (!files.length) {
      return [];
    }

    const documents: Document[] = [];

    for (const file of files) {
      try {
        const results = await this.provider.getDocsByPath(file.path);

        if (results.length > 0) {
          const docs = this.convertToDocuments(results, true);
          documents.push(...docs);
        } else {
          // If no results, try to read the file content directly
          const content = await this.app.vault.read(file);
          documents.push(
            new Document({
              pageContent: content,
              metadata: {
                path: file.path,
                title: file.basename,
              },
            })
          );
        }
      } catch (error) {
        logError(`Error getting document from file ${file.path}:`, error);
      }
    }

    return documents;
  }

  /**
   * Convert text to embedding
   * @param text The text to convert
   * @returns The embedding vector
   */
  private async convertTextToEmbedding(text: string): Promise<number[]> {
    try {
      const embeddingInstance = await this.embeddingsManager.getEmbeddingsAPI();
      const embeddings = await embeddingInstance.embedQuery(text);
      return embeddings;
    } catch (error) {
      logError("Error converting text to embedding:", error);
      throw new CustomError(`Failed to convert text to embedding: ${error.message}`);
    }
  }

  /**
   * Convert vector documents to langchain documents
   * @param results Array of VectorDocument objects
   * @param includeMetadata Whether to include metadata
   * @returns Array of Document objects
   */
  private convertToDocuments(results: VectorDocument[], includeMetadata: boolean): Document[] {
    return results.map((result) => {
      const metadata = includeMetadata
        ? {
            id: result.id,
            path: result.path,
            title: result.title,
            created_at: result.created_at,
            ctime: result.ctime,
            mtime: result.mtime,
            tags: result.tags,
            extension: result.extension,
            nchars: result.nchars,
            ...result.metadata,
          }
        : { path: result.path, title: result.title };

      return new Document({
        pageContent: result.content,
        metadata,
      });
    });
  }

  /**
   * Combine and deduplicate documents
   * @param documents1 First array of documents
   * @param documents2 Second array of documents
   * @returns Combined and deduplicated array of documents
   */
  private combineAndDeduplicate(documents1: Document[], documents2: Document[]): Document[] {
    const combined = [...documents1, ...documents2];
    const seen = new Set<string>();
    const deduplicated: Document[] = [];

    for (const doc of combined) {
      const path = doc.metadata.path as string;
      if (!seen.has(path)) {
        seen.add(path);
        deduplicated.push(doc);
      }
    }

    return deduplicated;
  }

  /**
   * Register a new provider type
   * @param providerType The provider type
   * @param constructor The provider constructor
   */
  public static registerProviderType(
    providerType: string,
    constructor: DBProviderConstructorType
  ): void {
    DBOperationsManager.providerMap[providerType] = {
      ProviderConstructor: constructor,
    };
  }

  /**
   * Check if a provider type is registered
   * @param providerType The provider type
   * @returns Whether the provider type is registered
   */
  public static hasProviderType(providerType: string): boolean {
    return providerType in DBOperationsManager.providerMap;
  }

  /**
   * Get all registered provider types
   * @returns Array of provider types
   */
  public static getProviderTypes(): string[] {
    return Object.keys(DBOperationsManager.providerMap);
  }

  /**
   * Ping a provider to check if it's available
   * @param providerType The provider type
   * @returns Whether the provider is available
   */
  public async pingProvider(providerType: string): Promise<boolean> {
    try {
      const provider = this.createProvider(providerType);
      const embeddingInstance = await this.embeddingsManager.getEmbeddingsAPI();
      await provider.initializeDB(embeddingInstance);
      return true;
    } catch (error) {
      logError(`Failed to ping provider ${providerType}:`, error);
      return false;
    }
  }
}
