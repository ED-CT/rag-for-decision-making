# RAG for Decision Making

This project has been created in the context of `Code Fol All` course of AI for Developers.
The goal if this PoC is to illustrate how one can filter and pulish the process used for decision-making in some particular contexts.

## Introduction to RAG
Retrieval-Augmented Generation (RAG) is a powerful technique that combines the strengths of large language models (LLMs) with efficient document retrieval techniques to provide context-aware and accurate responses to queries. It is particularly useful in applications where external knowledge needs to be integrated into the response generation process.

## LangChain Library
The **LangChain library** is an open-source framework designed to simplify the development of applications using large language models. It provides a modular environment that allows developers to integrate multiple LLMs into their applications seamlessly. LangChain supports both Python and JavaScript, making it versatile for various development needs. It facilitates the comparison of different prompts and models without requiring significant code changes, which is beneficial for optimizing LLM-driven applications like chatbots and virtual agents[2].

## Data Preparation and Data Cleaning
**Data preparation** is a crucial phase in building a RAG system. It involves loading data, organizing it into a consistent format, and breaking it down into smaller segments or chunks. This process ensures that the data is clean and structured, ready for embedding and retrieval. **Data cleaning** is an essential part of this phase, focusing on removing irrelevant or noisy information such as headers, footers, or special characters. This step helps reduce unnecessary data and improves the overall quality of the RAG system[3][4].

## Tokenization in LLMs
**Tokenization** is the process of breaking down text into smaller units called tokens. In the context of LLMs, tokenization is a preliminary step before embedding and processing. Tokenizers convert text into a format that can be understood by the model, allowing it to generate embeddings and process the input effectively. The choice of tokenizer can significantly impact the performance of the LLM, as it affects how the model interprets and processes the input text[5].

## Chunking
**Chunking** involves dividing large documents into smaller, manageable segments. This technique is essential in RAG systems for efficient retrieval and processing of relevant information. Chunking strategies can vary, including fixed-size chunking, recursive character text splitting, and semantic chunking. Each strategy has its advantages and is chosen based on the structure and requirements of the data being processed. Effective chunking enhances the accuracy and contextual awareness of AI-powered information retrieval and generation systems[6].

## Embedding
**Embeddings** are mathematical representations of objects, such as text chunks, that capture their semantic meaning. In RAG systems, embeddings are used to represent document chunks and user queries in a high-dimensional vector space. This allows for efficient vector searches to find the most relevant chunks based on their semantic similarity to the query. Embeddings are crucial for enabling the model to retrieve contextually relevant information from a large dataset[7].

## Vectorial Database
A **vectorial database** is used in RAG systems to store and manage the vector embeddings of document chunks. This database facilitates fast and efficient retrieval of relevant chunks based on their semantic similarity to a user's query. By leveraging vector search capabilities, RAG systems can quickly identify and retrieve the most relevant information from a vast knowledge base, enhancing the accuracy and contextuality of the generated responses[8].

## Inference in LLMs
**Inference in LLMs** refers to the process of using a trained language model to generate responses to user queries. In RAG systems, inference involves retrieving relevant chunks from the vector database and using these chunks as grounding data for the LLM. The model then generates a response based on this context, ensuring that the output is accurate and relevant to the query. This process combines the strengths of retrieval and generation to produce high-quality responses.

## Tools like Gradio
**Gradio** is a tool that can be used to build interactive interfaces for AI applications, including those based on RAG systems. While not specifically mentioned in the context of RAG, tools like Gradio can be used to create user-friendly interfaces for querying and visualizing results from RAG systems. However, for RAG implementations, **Streamlit** is more commonly used to create seamless user interfaces for document uploads, querying, and result visualization[1].

Citations:
[1] https://dev.to/ajmal_hasan/genai-building-rag-systems-with-langchain-4dbp
[2] https://www.ibm.com/think/topics/langchain
[3] https://learn.microsoft.com/en-us/azure/databricks/generative-ai/tutorials/ai-cookbook/quality-data-pipeline-rag
[4] https://annora.ai/blog/data-prep-rag
[5] https://cybernetist.com/2024/10/21/you-should-probably-pay-attention-to-tokenizers/
[6] https://www.f22labs.com/blogs/7-chunking-strategies-in-rag-you-need-to-know/
[7] https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-generate-embeddings
[8] https://objectbox.io/retrieval-augmented-generation-rag-with-vector-databases-expanding-ai-capabilities/
[9] https://milvus.io/ai-quick-reference/what-is-an-rag-retrievalaugmented-generation-vector-database
[10] https://snorkel.ai/large-language-models/rag-retrieval-augmented-generation/
[11] https://github.com/nsrinidhibhat/gradio_RAG
[12] https://www.linkedin.com/pulse/building-retrieval-augmented-generation-rag-agents-large-gaurav-kumar-zoolc
[13] https://github.com/agnedil/rag-demo-with-gradio
[14] https://pyimagesearch.com/2025/02/03/introduction-to-gradio-for-building-interactive-applications/
[15] https://www.recordlydata.com/blog/gradio-rag-and-vector-databases-where-to-start
[16] https://developer.ibm.com/articles/awb-retrieval-augmented-generation-with-langchain-and-elastic-db
[17] https://www.codemancers.com/blog/2024-10-24-rag-with-langchain/
[18] https://python.langchain.com/docs/concepts/rag/
[19] https://www.langchain.ca/blog/unlocking-the-power-of-retrieval-augmented-generation-rag-in-ai/
[20] https://python.langchain.com/docs/tutorials/rag/
[21] https://python.langchain.com/v0.2/docs/concepts/
[22] https://python.langchain.com/v0.1/docs/use_cases/question_answering/
[23] https://www.langchain.com
[24] https://www.codemag.com/Article/2501051/Exploring-LangChain-A-Practical-Approach-to-Language-Models-and-Retrieval-Augmented-Generation-RAG
[25] https://upstash.com/blog/langchain-explained
[26] https://mindit.io/blog/optimizing-data-retrieval-in-rag-systems
[27] https://www.youtube.com/watch?v=O0tuoGI2GH8
[28] https://www.couchbase.com/blog/guide-to-data-prep-for-rag/
[29] https://www.datasciencecentral.com/best-practices-for-structuring-large-datasets-in-retrieval-augmented-generation-rag/
[30] https://www.edlitera.com/blog/posts/retrieval-augmented-generation
[31] https://www.reddit.com/r/Rag/comments/1gu17gm/how_people_prepare_data_for_rag_applications/
[32] https://www.splunk.com/en_us/blog/learn/retrieval-augmented-generation-rag.html
[33] https://aws.amazon.com/what-is/retrieval-augmented-generation/
[34] https://www.appen.com/blog/enhancing-data-to-unlock-success-with-rag-optimization
[35] https://vectorize.io/i-built-a-rag-pipeline-from-scratch-heres-what-i-learned-about-unstructured-data/
[36] https://www.deepset.ai/blog/preprocessing-rag
[37] https://www.amazee.io/blog/post/data-pipelines-for-rag/
[38] https://christophergs.com/blog/understanding-llm-tokenization
[39] https://www.exxactcorp.com/blog/deep-learning/how-retrieval-augment-generation-makes-llms-smarter-than-before
[40] https://seantrott.substack.com/p/tokenization-in-large-language-models
[41] https://www.promptingguide.ai/research/rag
[42] https://cloud.google.com/use-cases/retrieval-augmented-generation
[43] https://www.youtube.com/watch?v=R9qAlrxbPK0
[44] https://www.reddit.com/r/learnmachinelearning/comments/1cs29kn/confused_about_embeddings_and_tokenization_in_llms/
[45] https://www.robertodiasduarte.com.br/compreendendo-tokens-e-retrieval-augmented-generation-rag-em-modelos-de-linguagem/
[46] https://www.promptingguide.ai/research/llm-tokenization
[47] https://www.youtube.com/watch?v=T_1pdsC0x0o
[48] https://www.datacamp.com/blog/what-is-tokenization
[49] https://bitpeak.com/chunking-methods-in-rag-methods-comparison/
[50] https://www.ai-bites.net/chunking-in-retrieval-augmented-generation-rag/
[51] https://developer.ibm.com/articles/awb-enhancing-rag-performance-chunking-strategies/
[52] https://stackoverflow.blog/2024/12/27/breaking-up-is-hard-to-do-chunking-in-rag-applications/
[53] https://www.sagacify.com/news/a-guide-to-chunking-strategies-for-retrieval-augmented-generation-rag
[54] https://towardsdatascience.com/rag-101-chunking-strategies-fdc6f6c2aaec/
[55] https://www.youtube.com/watch?v=v6g8eo86T8A
[56] https://www.mongodb.com/developer/products/atlas/choose-embedding-model-rag/
[57] https://qdrant.tech/articles/what-is-rag-in-ai/
[58] https://www.thecloudgirl.dev/blog/the-secret-sauce-of-rag-vector-search-and-embeddings
[59] https://www.galileo.ai/blog/mastering-rag-how-to-select-an-embedding-model
[60] https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/
[61] https://www.youtube.com/watch?v=A7qjjkjjZu0
[62] https://www.smashingmagazine.com/2024/01/guide-retrieval-augmented-generation-language-models/
[63] https://www.timescale.com/blog/finding-the-best-open-source-embedding-model-for-rag
[64] https://www.reddit.com/r/Rag/comments/1ezuzd7/understanding_embedding_models_make_an_informed/
[65] https://www.reddit.com/r/LocalLLaMA/comments/17qse19/rag_vs_vector_db/
[66] https://www.linkedin.com/pulse/vector-database-retrieval-augmented-generation-search-frank-fyzhc
[67] https://www.reddit.com/r/vectordatabase/comments/1hzovpy/best_vector_database_for_rag/
[68] https://www.youtube.com/watch?v=4l8zhHUBeyI
[69] https://github.com/microsoft/generative-ai-for-beginners/blob/main/15-rag-and-vector-databases/README.md?WT.mc_id=academic-105485-koreyst
[70] https://community.openai.com/t/best-vector-database-to-use-with-rag/615350
[71] https://www.youtube.com/watch?v=xS55duPS-Pw
[72] https://ethanlazuk.com/blog/inference-scaling-for-long-context-rag/
[73] https://techcommunity.microsoft.com/blog/startupsatmicrosoftblog/llm-rag-deploy-llm-inference-endpoints--optimize-output-with-rag/4222636
[74] https://arxiv.org/html/2405.16178v1
[75] https://developer.nvidia.com/blog/rag-101-demystifying-retrieval-augmented-generation-pipelines/
[76] https://linkml.io/linkml-store/how-to/Perform-RAG-Inference.html
[77] https://openreview.net/forum?id=FSjIrOm1vz
[78] https://towardsdatascience.com/retrieval-augmented-generation-rag-inference-engines-with-langchain-on-cpus-d5d55f398502/
[79] https://arxiv.org/abs/2410.20142
[80] https://www.snowflake.com/en/blog/easy-secure-llm-inference-retrieval-augmented-generation-rag-cortex/
[81] https://research.ibm.com/blog/retrieval-augmented-generation-RAG
[82] https://arxiv.org/html/2410.04343v1
[83] https://www.databricks.com/glossary/retrieval-augmented-generation-rag
[84] https://www.youtube.com/watch?v=Z-H-Uuq4d4E
[85] https://www.youtube.com/watch?v=hMC2iodrjCg
[86] https://ruqyai.github.io/posts/2024/04/blog-post-1/
[87] https://www.superteams.ai/blog/designing-a-rag-pipeline-with-mistral-weaviate-and-gradio-for-company-documents
[88] https://www.gradio.app/guides/creating-a-chatbot-fast
[89] https://upstash.com/docs/vector/tutorials/gradio-application
[90] https://www.youtube.com/watch?v=YLPNA1j7kmQ
[91] https://python.langchain.com/v0.2/docs/tutorials/rag/
[92] https://js.langchain.com/docs/introduction
[93] https://auth0.com/blog/building-a-secure-rag-with-python-langchain-and-openfga/
[94] https://www.pinecone.io/learn/series/langchain/langchain-intro/
[95] https://python.langchain.com/docs/introduction/
[96] https://www.vldb.org/pvldb/vol17/p4421-eltabakh.pdf
[97] https://towardsdatascience.com/a-guide-on-12-tuning-strategies-for-production-ready-rag-applications-7ca646833439/
[98] https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-preparation-phase
[99] https://llmstack.ai/blog/retrieval-augmented-generation
[100] https://community.databricks.com/t5/technical-blog/six-steps-to-improve-your-rag-application-s-data-foundation/ba-p/97700
[101] https://statusneo.com/all-you-need-to-know-about-retrieval-augmented-generationrag/
[102] https://docs.mistral.ai/guides/tokenization/
[103] https://airbyte.com/data-engineering-resources/llm-tokenization
[104] https://www.linkedin.com/posts/andrewyng_tokenization-turning-text-into-a-sequence-activity-7247262490340806657-f-Vv
[105] https://learn.microsoft.com/en-us/dotnet/ai/conceptual/understanding-tokens
[106] https://openreview.net/forum?id=tbx3u2oZAu
[107] https://arxiv.org/abs/2406.00944
[108] https://zilliz.com/learn/guide-to-chunking-strategies-for-rag
[109] https://www.ibm.com/think/tutorials/chunking-strategies-for-rag-with-langchain-watsonx-ai
[110] https://www.linkedin.com/pulse/chunking-retrieval-augmented-generation-rag-its-types-jashneet-kaur-mflic
[111] https://antematter.io/blogs/optimizing-rag-advanced-chunking-techniques-study
[112] https://www.mongodb.com/developer/products/atlas/choosing-chunking-strategy-rag/
[113] https://unstructured.io/blog/chunking-for-rag-best-practices
[114] https://milvus.io/ai-quick-reference/what-role-do-embeddings-play-in-rag-workflows
[115] https://www.aporia.com/learn/understanding-the-role-of-embeddings-in-rag-llms/
[116] https://wandb.ai/mostafaibrahim17/ml-articles/reports/Vector-Embeddings-in-RAG-Applications--Vmlldzo3OTk1NDA5
[117] https://blog.gopenai.com/boosting-rag-accuracy-part1-the-role-of-fine-tuning-an-embedding-model-for-domain-knowledge-279261b1bb22
[118] https://www.linkedin.com/pulse/rag-deep-dive-understanding-vector-embeddings-search-poornachandra-qdldf
[119] https://coralogix.com/ai-blog/understanding-the-role-of-embeddings-in-rag-llms/
[120] https://dataforest.ai/blog/vector-db-for-rag-information-retrieval-with-semantic-search
[121] https://apxml.com/posts/top-vector-databases-for-rag
[122] https://www.digitalocean.com/community/conceptual-articles/how-to-choose-the-right-vector-database
[123] https://writer.com/engineering/rag-vector-database/
[124] https://learn.microsoft.com/en-us/azure/databricks/generative-ai/tutorials/ai-cookbook/fundamentals-inference-chain-rag
[125] https://docs.getdynamiq.ai/low-code-builder/rag-nodes/inference-rag-workflow
[126] https://docs.databricks.com/aws/en/generative-ai/tutorials/ai-cookbook/fundamentals-inference-chain-rag
[127] https://www.redhat.com/en/topics/ai/what-is-retrieval-augmented-generation
[128] https://python.langchain.com/docs/integrations/tools/gradio_tools/
[129] https://www.gradio.app/guides/gradio-and-llm-agents
[130] https://www.datacamp.com/tutorial/deepseek-r1-rag

---
Core concepts explained with Perplexity.
