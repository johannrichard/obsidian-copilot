import { VECTORIZE_BATCH_LIMIT, METADATA_SAVE_INTERVAL } from "@/constants";
import { CustomError } from "@/error";
import { logError, logInfo } from "@/logger";
import { Embeddings } from "@langchain/core/embeddings";
import { App, Notice } from "obsidian";
import {
  DBProvider,
  VectorDocument,
  VectorDocumentMetadata,
  VectorSearchOptions,
} from "./dbProvider";
import { areEmbeddingModelsSame } from "@/utils";
import EmbeddingsManager from "@/LLMProviders/embeddingManager";
import { MD5 } from "crypto-js";
/**
 * Base class for cloud-based vector database providers
 * Implements common functionality that doesn't depend on specific API endpoints
 */
export abstract class BaseCloudDBProvider implements DBProvider {
  // Common state variables
  protected app: App;
  protected isInitialized = false;
  protected isInitializing = false;
  protected initializationError: Error | null = null;
  protected initializationPromise: Promise<void>;
  protected lastInitAttempt = 0;
  protected filesWithoutEmbeddings: Set<string> = new Set();
  protected pendingDocuments: VectorDocument[] = []; // In-memory store for pending documents
  protected metadataCache: Record<string, VectorDocumentMetadata> = {}; // Cache of document metadata
  protected metadataPath: string = ""; // Path to metadata file
  protected metadataLastSaved = 0;
  protected embeddingInstance: Embeddings;
  protected vectorLength: number = 0;
  protected embeddingModelName: string = "";
  protected collectionName: string;

  constructor(app: App) {
    this.app = app;
    // Set up metadata path – we need it in the config dir to be sure it syncs if sync is enabled
    this.metadataPath = `${this.app.vault.configDir}/copilot-clouddb-metadata-${this.getVaultIdentifier()}.json`;
    this.collectionName = this.getVaultIdentifier();
  }

  // Abstract methods that must be implemented by subclasses
  abstract initializeDB(embeddingInstance: Embeddings): Promise<void>;
  abstract saveDB(): Promise<void>;
  abstract recreateIndex(embeddingInstance: Embeddings): Promise<void>;
  abstract getDb(): void | undefined;

  // Implementation-specific methods that interact with the cloud API
  protected abstract deleteFromCloud(id: string): Promise<void>;
  protected abstract searchCloudByEmbedding(
    embedding: number[],
    limit: number,
    threshold: number
  ): Promise<VectorDocument[]>;
  protected abstract searchCloudByPath(path: string): Promise<VectorDocument[]>;
  protected abstract collectionExists(): Promise<boolean>;
  protected abstract createCollection(): Promise<void>;
  protected abstract clearCollection(): Promise<void>; // Clears the index (can be delete & recreating)
  protected abstract validateVectorDimensions(): Promise<boolean>;
  protected abstract getDocsFromCloudById(ids: string[]): Promise<VectorDocument[]>;

  public abstract checkIndexIntegrity(): Promise<void>;
  public abstract getDocsJsonByPaths(paths: string[]): Promise<Record<string, any[]>>;
  /**
   * Calculate cosine similarity between two vectors
   */
  protected calculateCosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) {
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

  /**
   * Create a schema for the vector database (for the sake of consistency)
   */
  public createSchema(vectorLength: number): any {
    // Base implementation - can be overridden by subclasses
    return {
      id: "string",
      path: "string",
      title: "string",
      content: "string",
      embedding: `vector[${vectorLength}]`,
      created_at: "number",
      ctime: "number",
      mtime: "number",
      tags: "string[]",
      extension: "string",
      nchars: "number",
      embeddingModel: "string",
    };
  }

  /**
   * Update the metadata cache with information from a document
   */
  protected updateMetadataCache(doc: VectorDocument): void {
    if (!doc.path) return;

    this.metadataCache[doc.id] = {
      id: doc.id,
      path: doc.path,
      embeddingModel: doc.embeddingModel || this.embeddingModelName,
      mtime: doc.mtime || Date.now(),
    };
  }

  /**
   * Remove a document from the metadata cache
   */
  protected removeFromMetadataCache(path: string): void {
    const idsToRemove = Object.entries(this.metadataCache)
      .filter(([_, metadata]) => metadata.path === path)
      .map(([id, _]) => id);

    for (const id of idsToRemove) {
      delete this.metadataCache[id];
    }
  }

  /**
   * Reinitialize the metadata cache
   */
  protected reinitializeMetadataCache(): boolean {
    try {
      this.metadataCache = {};
      this.metadataLastSaved = 0;
      this.saveMetadata();
      return true;
    } catch (error) {
      logError("Error clearing metadata cache:", error);
      return false;
    }
  }

  /**
   * Load metadata from disk if it exists
   */
  protected async loadMetadata(): Promise<void> {
    try {
      if (await this.app.vault.adapter.exists(this.metadataPath)) {
        const metadataJson = await this.app.vault.adapter.read(this.metadataPath);
        this.metadataCache = JSON.parse(metadataJson);
        logInfo(`Loaded metadata for ${Object.keys(this.metadataCache).length} documents`);
      } else {
        logInfo("No metadata file found, starting with empty cache");
        this.metadataCache = {};
      }
    } catch (error) {
      logError("Error loading metadata:", error);
      this.metadataCache = {};
    }
  }

  /**
   * Save metadata to disk if needed
   */
  protected async saveMetadata(): Promise<void> {
    try {
      // Only save if there have been changes and enough time has passed
      const now = Date.now();
      if (now - this.metadataLastSaved < METADATA_SAVE_INTERVAL) {
        return; // Throttle saves to avoid excessive disk writes
      }

      const metadataJson = JSON.stringify(this.metadataCache);
      await this.app.vault.adapter.write(this.metadataPath, metadataJson);
      this.metadataLastSaved = now;
      logInfo(`Saved metadata for ${Object.keys(this.metadataCache).length} documents`);
    } catch (error) {
      logError("Error saving metadata:", error);
    }
  }

  /**
   * Mark that there are unsaved changes
   */
  public markUnsavedChanges() {
    logInfo("Marking unsaved changes");
  }

  /**
   * Mark a file as missing embeddings
   */
  public markFileMissingEmbeddings(filePath: string): void {
    this.filesWithoutEmbeddings.add(filePath);
  }

  /**
   * Clear the list of files missing embeddings
   */
  public clearFilesMissingEmbeddings(): void {
    this.filesWithoutEmbeddings.clear();
  }

  /**
   * Get the list of files missing embeddings
   */
  public getFilesMissingEmbeddings(): string[] {
    return Array.from(this.filesWithoutEmbeddings);
  }

  /**
   * Check if a file is missing embeddings
   */
  public isFileMissingEmbeddings(filePath: string): boolean {
    return this.filesWithoutEmbeddings.has(filePath);
  }

  /**
   * Check if initialization is needed and possible, then attempt to initialize
   */
  protected async ensureInitialized(): Promise<boolean> {
    if (this.isInitialized) {
      return true;
    }

    // If we're already initializing, don't start another initialization
    if (this.isInitializing) {
      return false;
    }

    // If we've tried to initialize recently and failed, don't try again too soon
    const now = Date.now();
    if (this.initializationError && now - this.lastInitAttempt < 30000) {
      return false;
    }

    // Try to initialize
    try {
      await this.initializeDB(this.embeddingInstance);
      return this.isInitialized;
    } catch (error) {
      logError("Failed to initialize during retry:", error);
      return false;
    }
  }

  /**
   * Manually trigger a retry of initialization
   */
  public async retryInitialization(): Promise<boolean> {
    if (this.isInitialized) {
      return true;
    }

    if (this.isInitializing) {
      return false;
    }

    try {
      await this.initializeDB(this.embeddingInstance);
      return this.isInitialized;
    } catch (error) {
      logError("Failed to initialize during manual retry:", error);
      return false;
    }
  }

  /**
   * Wait for initialization to complete
   */
  public async waitForInitialization(): Promise<void> {
    await this.initializationPromise;
  }

  /**
   * Get the list of indexed files
   */
  public async getIndexedFiles(): Promise<string[]> {
    try {
      await this.ensureInitialized();

      // Get paths from metadata cache
      const metadataPaths = Object.values(this.metadataCache).map((metadata) => metadata.path);

      // Get paths from pending documents that aren't in metadata yet
      const pendingPaths = this.pendingDocuments
        .map((doc) => doc.path)
        .filter((path) => !metadataPaths.includes(path));

      return [...metadataPaths, ...pendingPaths];
    } catch (err) {
      logError("Failed to retrieve indexed files:", err);
      throw new CustomError("Failed to retrieve indexed files.");
    }
  }

  /**
   * Check if a note has been indexed
   */
  public async hasIndex(notePath: string): Promise<boolean> {
    // Check Metadata cache
    const docs = Object.values(this.metadataCache);
    const doc = docs.filter((doc) => {
      return doc.path === notePath;
    });

    return Promise.resolve(doc.length > 0);
  }

  /**
   * Check if a file is indexed based on metadata
   */
  public async isFileIndexed(path: string): Promise<boolean> {
    if (!path) return false;

    try {
      // First check the pending documents
      const hasPendingDoc = this.pendingDocuments.some((doc) => doc.path === path);
      if (hasPendingDoc) {
        return true;
      }

      // Then check metadata cache
      if (Object.values(this.metadataCache).some((metadata) => metadata.path === path)) {
        return true;
      }

      // If not found in memory, it's not indexed
      return false;
    } catch (error) {
      logError(`Error checking if file ${path} is indexed:`, error);
      return false;
    }
  }

  /**
   * Get all indexed file paths
   */
  public async getAllIndexedFilePaths(): Promise<string[]> {
    const metadataPaths = Object.values(this.metadataCache).map((metadata) => metadata.path);

    // Get paths from pending documents that aren't in metadata yet
    const pendingPaths = this.pendingDocuments
      .map((doc) => doc.path)
      .filter((path) => !metadataPaths.includes(path));

    return [...metadataPaths, ...pendingPaths];
  }

  /**
   * Get the latest modification time of any indexed file
   */
  public async getLatestFileMtime(): Promise<number> {
    try {
      let latestMtime = 0;

      // Check metadata cache
      for (const metadata of Object.values(this.metadataCache)) {
        const mtime = metadata.mtime || 0;
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
      logError("Error getting latest file mtime:", error);
      return 0;
    }
  }

  /**
   * Check if the index is empty
   */
  public async isIndexEmpty(): Promise<boolean> {
    // If we have pending documents or metadata, the index is not empty
    logInfo("Checking if index is empty");
    if (this.pendingDocuments.length > 0 || Object.keys(this.metadataCache).length > 0) {
      return false;
    }

    // Check the cloud index
    try {
      return await this.collectionExists();
    } catch (error) {
      logError("Error checking if index is empty:", error);
      return true; // Assume empty on error
    }
  }

  /**
   * Check if a document exists in the database
   */
  public async hasDocument(path: string): Promise<boolean> {
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

      // If we're not initialized, we can't check the cloud
      if (!this.isInitialized) {
        return false;
      }

      // Check the cloud database
      const docs = await this.getDocsByPath(path);
      return docs.length > 0;
    } catch (error) {
      logError(`Error checking if document ${path} exists:`, error);
      return false;
    }
  }

  /**
   * Check if a document has embeddings
   */
  public async hasEmbeddings(path: string): Promise<boolean> {
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

    // If we're not initialized, we can't check the cloud
    if (!this.isInitialized) {
      return false;
    }

    // Check the cloud database
    try {
      const docs = await this.getDocsByPath(path);

      if (!docs || docs.length === 0) {
        return false;
      }

      // Check if all vectors have values
      return docs.every(
        (doc: any) => doc.embedding && Array.isArray(doc.embedding) && doc.embedding.length > 0
      );
    } catch (error) {
      logError(`Error checking embeddings for path ${path}:`, error);
      return false;
    }
  }

  /**
   * Handle cleanup when the plugin is unloaded
   */
  public async onunload(): Promise<void> {
    // Save any pending documents before shutdown
    if (this.isInitialized && this.pendingDocuments.length > 0) {
      try {
        await this.saveDB();
      } catch (error) {
        logError("Error saving DB during unload:", error);
      }
    }
  }

  /**
   * Perform garbage collection by removing documents that no longer exist
   */
  public async garbageCollect(): Promise<number> {
    try {
      // Get all indexed file paths
      const indexedPaths = await this.getAllIndexedFilePaths();

      // Check which files still exist
      const pathsToRemove: string[] = [];

      for (const path of indexedPaths) {
        try {
          const exists = await this.app.vault.adapter.exists(path);
          if (!exists) {
            pathsToRemove.push(path);
          }
        } catch (error) {
          logError(`Error checking if file ${path} exists:`, error);
        }
      }

      logInfo(`Found ${pathsToRemove.length} files to remove during garbage collection`);

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
   * Get all documents from the database
   */
  public async getAllDocuments(): Promise<VectorDocument[]> {
    try {
      // Try to ensure we're initialized, but don't block if not
      await this.ensureInitialized();

      // Start with pending documents
      const documents: VectorDocument[] = [...this.pendingDocuments];

      // Add documents from metadata cache
      for (const [id, metadata] of Object.entries(this.metadataCache)) {
        // Skip documents that are already in the pending documents
        if (documents.some((doc) => doc.id === id)) {
          continue;
        }

        // Create a minimal document from metadata
        const doc: VectorDocument = {
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
          metadata: metadata,
        };

        documents.push(doc);
      }

      logInfo(`Retrieved ${documents.length} documents from metadata cache and pending documents`);
      return documents;
    } catch (error) {
      logError("Error getting all documents from metadata cache:", error);
      return this.pendingDocuments; // Return at least the pending documents if metadata retrieval fails
    }
  }

  /**
   * Check if the index is loaded
   */
  public async getIsIndexLoaded(): Promise<boolean> {
    return this.isInitialized;
  }

  /**
   * Get the current database path
   */
  public getCurrentDbPath(): string {
    // To be implemented by subclasses
    return "";
  }

  /**
   * Get the database path
   */
  public async getDbPath(): Promise<string> {
    // To be implemented by subclasses
    return this.getCurrentDbPath();
  }

  /**
   * Get the storage path description
   */
  public async getStoragePath(): Promise<string> {
    // To be implemented by subclasses
    return "Cloud DB Provider";
  }

  /**
   * Check and handle embedding model changes
   */
  public async checkAndHandleEmbeddingModelChange(embeddingInstance: Embeddings): Promise<boolean> {
    if (!this.isInitialized) {
      await this.initializeDB(embeddingInstance);
    }

    // this.initializeEmbeddingModel(embeddingInstance);

    let prevEmbeddingModel: string | undefined;

    const singleDoc = Object.values(this.metadataCache)[0];
    if (singleDoc && singleDoc.embeddingModel) {
      prevEmbeddingModel = singleDoc.embeddingModel;
    }

    if (prevEmbeddingModel) {
      const currEmbeddingModel = EmbeddingsManager.getModelName(embeddingInstance);

      if (!areEmbeddingModelsSame(prevEmbeddingModel, currEmbeddingModel)) {
        // Model has changed, notify user and rebuild DB
        new Notice("New embedding model detected. Rebuilding Copilot index from scratch.");
        logInfo("Detected change in embedding model. Rebuilding Copilot index from scratch.");

        // Create new DB with new model, clear metadata cache
        await this.recreateIndex(embeddingInstance);
        return true;
      }
    } else {
      logInfo("No previous embedding model found in the database.");
    }

    return Promise.resolve(false);
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

  /**
   * Remove documents from the database
   */
  public async removeDocs(path: string): Promise<void> {
    try {
      // Remove from pending documents
      const pendingDocsToRemove = this.pendingDocuments.filter((doc) => doc.path === path);
      if (pendingDocsToRemove.length > 0) {
        this.pendingDocuments = this.pendingDocuments.filter((doc) => doc.path !== path);
        logInfo(
          `Removed ${pendingDocsToRemove.length} documents with path ${path} from pending documents`
        );
      }

      // Remove from metadata cache
      this.removeFromMetadataCache(path);

      // If not initialized, we can't remove from the cloud
      if (!this.isInitialized) {
        logInfo("Cloud DB provider not initialized, only removing from local cache");
        return;
      }

      // Get document IDs from metadata cache
      const idsToRemove = Object.entries(this.metadataCache)
        .filter(([_, metadata]) => metadata.path === path)
        .map(([id, _]) => id);

      // Delete from cloud
      for (const id of idsToRemove) {
        await this.deleteFromCloud(id);
      }

      logInfo(`Removed ${idsToRemove.length} documents with path: ${path} from Cloud DB`);

      // Save metadata after removal
      await this.saveMetadata();
    } catch (error) {
      logError(`Error removing documents with path ${path} from Cloud DB:`, error);
    }
  }

  /**
   * Get documents by path
   */
  public async getDocsByPath(path: string): Promise<VectorDocument[]> {
    // Try to ensure we're initialized, but don't block if not
    await this.ensureInitialized();

    if (!this.isInitialized) {
      throw new CustomError("Cloud DB provider not initialized");
    }

    try {
      // First check pending documents
      const pendingDocs = this.pendingDocuments.filter((doc) => doc.path === path);

      // // Get document IDs from metadata cache
      // const ids = Object.entries(this.metadataCache)
      //   .filter(([_, metadata]) => metadata.path === path)
      //   .map(([id, _]) => id);

      // // Get documents from cloud
      // const dbDocs = await this.getDocsFromCloudById(ids);

      const dbDocs = await this.searchCloudByPath(path);
      const results: VectorDocument[] = [...pendingDocs, ...dbDocs];

      // If we found documents in the cache or pending documents, return them
      if (results.length > 0) {
        logInfo(`Found ${results.length} documents for path ${path}`);
        return results;
      }
      return [];
    } catch (error) {
      logError(`Error getting documents for path ${path}:`, error);
      return [];
    }
  }

  /**
   * Get documents by embedding
   */
  public async getDocsByEmbedding(
    embedding: number[],
    options: VectorSearchOptions
  ): Promise<VectorDocument[]> {
    // Try to ensure we're initialized
    await this.ensureInitialized();

    if (!this.isInitialized) {
      throw new CustomError("Cloud DB provider not initialized");
    }

    try {
      // Search the cloud database
      const cloudResults = await this.searchCloudByEmbedding(
        embedding,
        options.limit,
        options.similarity
      );

      // If there are no pending documents, return the cloud results
      if (this.pendingDocuments.length === 0) {
        return cloudResults;
      }

      // Calculate similarity for pending documents
      const pendingResults = this.pendingDocuments
        .map((doc) => {
          if (!doc.embedding || doc.embedding.length === 0) {
            return { ...doc, similarity: 0 };
          }

          const similarity = this.calculateCosineSimilarity(embedding, doc.embedding);
          return { ...doc, similarity };
        })
        .filter((doc) => (doc as any).similarity >= options.similarity);

      // Combine and sort results
      const combinedResults = [...cloudResults, ...pendingResults]
        .sort((a, b) => {
          const aSimilarity = "similarity" in a ? ((a as any).similarity as number) : 0;
          const bSimilarity = "similarity" in b ? ((b as any).similarity as number) : 0;
          return bSimilarity - aSimilarity;
        })
        .slice(0, options.limit); // Apply limit

      return combinedResults;
    } catch (error) {
      logError("Error searching for documents by embedding:", error);
      return [];
    }
  }
  /**
   * Gets a unique identifier for the vault
   * This is used to create a consistent filename for the metadata cache
   */
  protected getVaultIdentifier(): string {
    const vaultName = this.app.vault.getName();
    return MD5(vaultName).toString();
  }
}
