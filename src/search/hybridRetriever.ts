import { BrevilabsClient } from "@/LLMProviders/brevilabsClient";
import ChatModelManager from "@/LLMProviders/chatModelManager";
import EmbeddingManager from "@/LLMProviders/embeddingManager";
import { logInfo } from "@/logger";
import { getSettings } from "@/settings/model";
import { extractNoteFiles, removeThinkTags } from "@/utils";
import { BaseCallbackConfig } from "@langchain/core/callbacks/manager";
import { Document } from "@langchain/core/documents";
import { BaseChatModelCallOptions } from "@langchain/core/language_models/chat_models";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { BaseRetriever } from "@langchain/core/retrievers";
import { TFile } from "obsidian";
import { VectorSearchService } from "./vectorSearchService";

export class HybridRetriever extends BaseRetriever {
  public lc_namespace = ["hybrid_retriever"];

  private queryRewritePrompt: ChatPromptTemplate;

  constructor(
    private options: {
      minSimilarityScore: number;
      maxK: number;
      salientTerms: string[];
      timeRange?: { startTime: number; endTime: number };
      textWeight?: number;
      returnAll?: boolean;
      useRerankerThreshold?: number; // reranking API is only called with this set
    }
  ) {
    super();
    this.queryRewritePrompt = ChatPromptTemplate.fromTemplate(
      "Please write a passage to answer the question. If you don't know the answer, just make up a passage. \nQuestion: {question}\nPassage:"
    );
  }

  public async getRelevantDocuments(
    query: string,
    config?: BaseCallbackConfig
  ): Promise<Document[]> {
    // Extract note TFiles wrapped in [[]] from the query
    const noteFiles = extractNoteFiles(query, app.vault);
    // Add note titles to salient terms
    const noteTitles = noteFiles.map((file) => file.basename);
    // Use Set to ensure uniqueness when combining terms
    const enhancedSalientTerms = [...new Set([...this.options.salientTerms, ...noteTitles])];

    // Retrieve chunks for explicitly mentioned note files
    const explicitChunks = await this.getExplicitChunks(noteFiles);
    let rewrittenQuery = query;
    if (config?.runName !== "no_hyde") {
      // Use config to determine if HyDE should be used
      // Generate a hypothetical answer passage
      rewrittenQuery = await this.rewriteQuery(query);
    }
    // Pass enhanced salient terms to include titles
    const vectorChunks = await this.getVectorChunks(
      rewrittenQuery,
      enhancedSalientTerms,
      this.options.textWeight
    );

    const combinedChunks = this.filterAndFormatChunks(vectorChunks, explicitChunks);

    let finalChunks = combinedChunks;

    // Add check for empty array
    if (combinedChunks.length === 0) {
      if (getSettings().debug) {
        console.log("No chunks found for query:", query);
      }
      return finalChunks;
    }

    const maxOramaScore = combinedChunks.reduce((max, chunk) => {
      const score = chunk.metadata.score;
      const isValidScore = typeof score === "number" && !isNaN(score);
      return isValidScore ? Math.max(max, score) : max;
    }, 0);

    const allScoresAreNaN = combinedChunks.every(
      (chunk) => typeof chunk.metadata.score !== "number" || isNaN(chunk.metadata.score)
    );

    const shouldRerank =
      this.options.useRerankerThreshold &&
      (maxOramaScore < this.options.useRerankerThreshold || allScoresAreNaN);
    // Apply reranking if max score is below the threshold or all scores are NaN
    if (shouldRerank) {
      const rerankResponse = await BrevilabsClient.getInstance().rerank(
        query,
        // Limit the context length to 3000 characters to avoid overflowing the reranker
        combinedChunks.map((doc) => doc.pageContent.slice(0, 3000))
      );

      // Map chunks based on reranked scores and include rerank_score in metadata
      finalChunks = rerankResponse.response.data.map((item) => ({
        ...combinedChunks[item.index],
        metadata: {
          ...combinedChunks[item.index].metadata,
          rerank_score: item.relevance_score,
        },
      }));
    }

    if (getSettings().debug) {
      console.log("*** HYBRID RETRIEVER DEBUG INFO: ***");

      if (config?.runName !== "no_hyde") {
        console.log("\nOriginal Query: ", query);
        console.log("Rewritten Query: ", rewrittenQuery);
      }

      console.log("\nExplicit Chunks: ", explicitChunks);
      console.log("Vector Chunks: ", vectorChunks);
      console.log("Combined Chunks: ", combinedChunks);
      console.log("Max Orama Score: ", maxOramaScore);
      if (shouldRerank) {
        console.log("Reranked Chunks: ", finalChunks);
      } else {
        console.log("No reranking applied.");
      }
    }

    return finalChunks;
  }

  private async rewriteQuery(query: string): Promise<string> {
    try {
      const promptResult = await this.queryRewritePrompt.format({ question: query });
      const chatModel = ChatModelManager.getInstance()
        .getChatModel()
        .bind({ temperature: 0 } as BaseChatModelCallOptions);
      const rewrittenQueryObject = await chatModel.invoke(promptResult);

      // Directly return the content assuming it's structured as expected
      if (rewrittenQueryObject && "content" in rewrittenQueryObject) {
        return removeThinkTags(rewrittenQueryObject.content as string);
      }
      console.warn("Unexpected rewrittenQuery format. Falling back to original query.");
      return query;
    } catch (error) {
      console.error("Error in rewriteQuery:", error);
      // If there's an error, return the original query
      return query;
    }
  }

  private async getExplicitChunks(noteFiles: TFile[]): Promise<Document[]> {
    if (noteFiles.length === 0) {
      return [];
    }

    const explicitChunks: Document[] = [];

    // Use VectorSearchService to get documents by path
    const searchService = VectorSearchService.getInstance(app);

    for (const file of noteFiles) {
      try {
        const docs = await searchService.searchByPath(file.path);
        if (docs.length > 0) {
          // Add includeInContext flag to metadata
          const docsWithContext = docs.map((doc) => ({
            ...doc,
            metadata: {
              ...doc.metadata,
              includeInContext: true,
            },
          }));
          explicitChunks.push(...docsWithContext);
        }
      } catch (error) {
        console.error(`Error getting chunks for file ${file.path}:`, error);
      }
    }

    return explicitChunks;
  }

  public async getVectorChunks(
    query: string,
    salientTerms: string[],
    textWeight?: number
  ): Promise<Document[]> {
    try {
      // Use the VectorSearchService instead of direct Orama search
      const searchService = VectorSearchService.getInstance(app);

      // Set up search options
      const searchOptions = {
        limit: this.options.maxK,
        similarity: this.options.minSimilarityScore,
        includeMetadata: true,
        salientTerms: salientTerms,
      };

      // If we have a time range, we need to handle it separately
      if (this.options.timeRange) {
        const { startTime, endTime } = this.options.timeRange;

        // Generate daily note date range
        const dailyNotes = this.generateDailyNoteDateRange(startTime, endTime);
        logInfo(
          "==== Daily note date range: ====",
          dailyNotes[0],
          dailyNotes[dailyNotes.length - 1]
        );

        // Get documents for daily notes
        const dailyNoteFiles = extractNoteFiles(dailyNotes.join(", "), app.vault);
        const dailyNoteResults = await this.getExplicitChunks(dailyNoteFiles);

        // Set includeInContext to true for all dailyNoteResults
        const dailyNoteResultsWithContext = dailyNoteResults.map((doc) => ({
          ...doc,
          metadata: {
            ...doc.metadata,
            includeInContext: true,
          },
        }));

        // Perform the search
        const searchResults = await searchService.searchByText(query, searchOptions);

        // Filter results by time range
        const timeFilteredResults = searchResults.filter(
          (doc) => doc.metadata.mtime >= startTime && doc.metadata.mtime <= endTime
        );

        // Combine and deduplicate results
        const combinedResults = [...dailyNoteResultsWithContext, ...timeFilteredResults];
        const uniqueResults = Array.from(
          new Set(combinedResults.map((doc) => (doc.metadata as any).id))
        ).map((id) => combinedResults.find((doc) => (doc.metadata as any).id === id));

        return uniqueResults.filter((doc): doc is Document => doc !== undefined);
      }

      // For regular searches without time range
      return await searchService.searchByText(query, searchOptions);
    } catch (error) {
      console.error("Error in getVectorChunks:", error);
      throw error;
    }
  }

  private async convertQueryToVector(query: string): Promise<number[]> {
    const embeddingsAPI = await EmbeddingManager.getInstance().getEmbeddingsAPI();
    const vector = await embeddingsAPI.embedQuery(query);
    if (vector.length === 0) {
      throw new Error("Query embedding returned an empty vector");
    }
    return vector;
  }

  private generateDailyNoteDateRange(startTime: number, endTime: number): string[] {
    const dailyNotes: string[] = [];
    const start = new Date(startTime);
    const end = new Date(endTime);

    const current = new Date(start);
    while (current <= end) {
      dailyNotes.push(`[[${current.toLocaleDateString("en-CA")}]]`);
      current.setDate(current.getDate() + 1);
    }

    return dailyNotes;
  }

  private filterAndFormatChunks(vectorChunks: Document[], explicitChunks: Document[]): Document[] {
    const threshold = this.options.minSimilarityScore;
    // Only filter out scores that are numbers and below threshold
    const filteredVectorChunks = vectorChunks.filter((chunk) => {
      const score = chunk.metadata.score;
      if (typeof score !== "number" || isNaN(score)) {
        return true; // Keep chunks with NaN scores for now until we find out why
      }
      return score >= threshold;
    });

    // Combine explicit and filtered Vector chunks, removing duplicates while maintaining order
    const uniqueChunks = new Set<string>(explicitChunks.map((chunk) => chunk.pageContent));
    const combinedChunks: Document[] = [...explicitChunks];

    for (const chunk of filteredVectorChunks) {
      const chunkContent = chunk.pageContent;
      if (!uniqueChunks.has(chunkContent)) {
        uniqueChunks.add(chunkContent);
        combinedChunks.push(chunk);
      }
    }

    // Add a new metadata field to indicate if the chunk should be included in the context
    return combinedChunks.map((chunk) => ({
      ...chunk,
      metadata: {
        ...chunk.metadata,
        includeInContext: true,
      },
    }));
  }
}
