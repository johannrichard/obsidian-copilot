import { Embeddings } from "@langchain/core/embeddings";
import { Orama } from "@orama/orama";

/*
 * Core metadata interface for vector databases
 */
export interface VectorDocumentMetadata {
  id: string;
  path: string;
  mtime: number;
  embeddingModel: string;
}

/**
 * Core document interface for vector databases
 */
export interface VectorDocument {
  id: string;
  title: string;
  content: string;
  embedding: number[];
  path: string;
  embeddingModel: string;
  created_at: number;
  ctime: number;
  mtime: number;
  tags: string[];
  extension: string;
  nchars: number;
  metadata: Record<string, any>;
}

/**
 * Search options for vector similarity search
 */
export interface VectorSearchOptions {
  limit: number;
  similarity: number;
}

/**
 * Base abstract class for vector database providers
 */
export abstract class DBProvider {
  /**
   * Initialize the database with an embeddings instance
   */
  abstract initializeDB(
    embeddingInstance: Embeddings | undefined
  ): Promise<Orama<any> | void | undefined>;

  /**
   * Save the current state of the database
   */
  abstract saveDB(): Promise<void>;

  /**
   * Clear the entire index and create a new empty database
   */
  abstract recreateIndex(embeddingInstance: Embeddings | undefined): Promise<void>;

  /**
   * Remove documents from the database by file path
   */
  abstract removeDocs(filePath: string): Promise<void>;

  /**
   * Get the current database instance
   */
  abstract getDb(): Orama<any> | void | undefined;

  /**
   * Check if the index is loaded
   */
  abstract getIsIndexLoaded(): Promise<boolean>;

  /**
   * Wait for the database to be fully initialized
   */
  abstract waitForInitialization(): Promise<void>;

  /**
   * Clean up resources when unloading
   */
  abstract onunload(): void;

  /**
   * Get the current database path
   */
  abstract getCurrentDbPath(): string;

  /**
   * Get the database path according to current settings
   */
  abstract getDbPath(): Promise<string>;

  /**
   * Mark that the database has unsaved changes
   */
  abstract markUnsavedChanges(): void;

  /**
   * Insert or update a document in the database
   */
  abstract upsert(docToSave: any, commitToDb: boolean): Promise<any | undefined>;

  /**
   * Check for embedding model changes and handle accordingly
   */
  abstract checkAndHandleEmbeddingModelChange(embeddingInstance: Embeddings): Promise<boolean>;

  /**
   * Remove documents that no longer exist in the vault
   */
  abstract garbageCollect(): Promise<number>;

  /**
   * Get a list of all indexed file paths
   */
  abstract getIndexedFiles(): Promise<string[]>;

  /**
   * Check if the index is empty
   */
  abstract isIndexEmpty(): Promise<boolean>;

  /**
   * Check if a specific note path is indexed
   */
  abstract hasIndex(notePath: string): Promise<boolean>;

  /**
   * Check if a specific note has embeddings
   */
  abstract hasEmbeddings(notePath: string): Promise<boolean>;

  /**
   * Get document JSON data by file paths
   */
  abstract getDocsJsonByPaths(paths: string[]): Promise<Record<string, any[]>>;

  /**
   * Mark a file as missing embeddings
   */
  abstract markFileMissingEmbeddings(filePath: string): void;

  /**
   * Clear the list of files missing embeddings
   */
  abstract clearFilesMissingEmbeddings(): void;

  /**
   * Get the list of files missing embeddings
   */
  abstract getFilesMissingEmbeddings(): string[];

  /**
   * Check if a file is missing embeddings
   */
  abstract isFileMissingEmbeddings(filePath: string): boolean;

  /**
   * Check the integrity of the index
   */
  abstract checkIndexIntegrity(): Promise<void>;

  /**
   * Get the latest file modification time from the database
   */
  abstract getLatestFileMtime(): Promise<number>;

  /**
   * Get documents by path from the database
   */
  abstract getDocsByPath(path: string): Promise<VectorDocument[]>;

  /**
   * Get documents by embedding vector
   */
  abstract getDocsByEmbedding(
    embedding: number[],
    options: {
      limit: number;
      similarity: number;
    }
  ): Promise<any[]>;

  /**
   * Get all documents from the database
   */
  abstract getAllDocuments(): Promise<any[]>;
}
