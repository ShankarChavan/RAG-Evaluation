# RAG-Evaluation
Topics we will cover 
* Basic Retrieval Augmented Generation workflow.
* Why we need to evaluate RAG applications.
* Types of evaluating metrics with code examples.
* Test-Data Generation for given document source
* Unit Testing RAG in CI/CD Pipelines 


## Overview of Metrics
A metric is a quantitative measure used to evaluate the performance of a AI application.
![alt text](images/image.png)

**LLM-based metrics** : 
These metrics uses `LLM` underneath to do the evaluation.

```python
from ragas.metrics import FactualCorrectness
scorer = FactualCorrectness(llm=evaluation_llm)
```
Each LLM based metrics also will have prompts associated with it written using Prompt Object.

**Non-LLM-based metrics**: 
These metrics `do not use LLM` underneath to do the evaluation.Instead uses string similarity and BLEU score etc.Hence these metrics are known to have a lower correlation with human evaluation.

**Single turn metrics**: It's a single turn interaction between Human and AI
```python
from ragas.metrics import FactualCorrectness

scorer = FactualCorrectness()
await scorer.single_turn_ascore(sample)
```

**Multi turn metrics**: It's a Multi turn interaction between Human and AI
```python
from ragas.metrics import AgentGoalAccuracy
from ragas import MultiTurnSample

scorer = AgentGoalAccuracy()
await scorer.multi_turn_ascore(sample)
```

## Basic RAG workflow

The idea behind the RAG framework is to combine a retrieval model and a generative model.

![Basic RAG workflow - Diagram by Author](https://assets.super.so/88507ae9-2dbb-4d4b-a38a-5cbaf10b19ff/images/27e1ff37-b626-4d61-9cbe-e8489a12247d/Untitled_design_(15).gif?w=708)

### Indexing

Indexing is the first important step in preparing data for language models. The original data is cleaned, converted to plain text, and broken into smaller chunks for easier processing.

These chunks are then turned into vector representations using an embedding model, which helps compare similarities during retrieval. The final index stores these text chunks and their vector embeddings, allowing for efficient and scalable searches.

### Retrieval

When a user asks a question, the system uses the encoding model from the indexing phase to convert the question into a vector. It then calculates similarity scores between this vector and the vectorized chunks in the index.

The system retrieves the top K chunks with the highest similarity scores, which are used to provide context for the user’s request.

### Generation

The user’s question and the selected documents are combined to create a clear prompt for a large language model. The model then crafts a response, adjusting its approach based on the specific task.

Now that we have a basic understanding of how the RAG workflow operates, let's move on to the evaluation phase.

Why we need to **evaluate** RAG applications?
--------------------------------------------

Evaluating RAG applications is important for understanding how well these pipelines work. We can see how effectively they combine information retrieval with generative models by checking their accuracy and relevance.

This evaluation helps improve RAG applications in tasks like 
- text summarization tasks, 
- chatbots,
- question-answering. 

It also identifies areas for improvement, ensuring that these systems provide trustworthy responses as information changes.

Overall, effective evaluation helps to 
- optimize performance and 
- builds confidence 

in RAG applications for real-world use.

How to Evaluate RAG applications?
---------------------------------

To evaluate a RAG application we focus on these main elements:

*   **Retrieval**: Experiment with various data processing strategies and embedding models to see how they affect retrieval performance.
*   **Generation**: After selecting the best settings for retrieval, test different large language models (LLMs) to find the best model for generating completions for the task.

To evaluate these elements, we will focus on the **key metrics** commonly used in RAG evaluation:

*   Context Precision
*   Context Recall
*   Answer Relevancy
*   Faithfulness

Let's break down each of the evaluation metrics one by one.

Context Precision
-----------------

Context Precision evaluates how well a retrieval system ranks the relevant pieces/chunks of information compared to the ground truth.

This metric is calculated using the query, ground truth, and context. The scores range from 0 to 1, with higher scores showing better precision.

### Formula:

![](https://hub.athina.ai/content/images/2024/11/1c3066a837d85f27911166d96812c718-1.png)

![](https://hub.athina.ai/content/images/2024/11/2-1.png)

k = retrieved chunks (or contexts) are relevant to the task

K = The total number of chunks in the retrieved contexts

### Example:

Consider we have 3 different examples which are in list format:

**Questions** = What is SpaceX?, Who found it?, What exactly does SpaceX do?

**Answers** = It is an American aerospace company, SpaceX founded by Elon Musk, SpaceX produces and operates the Falcon 9 and Falcon rockets

**Contexts** = SpaceX is an American aerospace company founded in 2002, SpaceX founded by Elon Musk, is worth nearly $210 billion, The full form of SpaceX is Space Exploration Technologies Corporation

**Ground Truth** = SpaceX is an American aerospace company, Founded by Elon Musk, SpaceX produces and operates the Falcon 9 and Falcon Heavy rockets

### Solution:

**For Question 1:** The **Ground Truth** is relevant to the **Context.** So, this is a true positive (TP), and there are no false positives (FP). Therefore, the context precision here is **1**.

![](https://hub.athina.ai/content/images/2024/11/3.png)

Similarly**,** for **Question 2,** the context precision is **1**.

But,

**For Question 3:** The **Ground Truth** is not relevant to the **Context.** Therefore, this is a false positive (FP), with no true positives (TP). Thus, the context precision here is **0**.

![](https://hub.athina.ai/content/images/2024/11/4.png)

For the average contextual precision with K=3, we assume equal weights Vk​=1, so our final answer becomes:

![](https://hub.athina.ai/content/images/2024/11/5.png)

Context Recall
--------------

Context recall measures how many relevant documents or pieces of information were retrieved. It helps evaluate if the retrieved context includes the key facts from the ground truth.

The score ranges from 0 to 1, where higher values indicate better performance.

### Formula:

![](https://hub.athina.ai/content/images/2024/11/1-1.png)

### Example:

**Question** = What is SpaceX and Who found it?

**Answer** = It is an American aerospace company founded by Elon Musk

**Context** = SpaceX is an American aerospace company founded in 2002.

**Ground Truth** = SpaceX is an American aerospace company founded by Elon Musk.

### Solution:

First, break down the sentence from Ground truth and check them against the context.

**Statement 1:** "SpaceX is an American aerospace company" (Yes, it is in the context)

**Statement 2:** "Founded by Elon Musk" (No, this part is not in the context)

In this example, we have 1st statement that is present in the context, while the 2nd statement is not present in the context.

Therefore, the context recall here is:

![](https://hub.athina.ai/content/images/2024/11/2-2.png)



Response(Answer) Relevancy
----------------

Response Relevancy is calculated as the average **cosine similarity** between the original question and a set of generated questions, where the generated questions are created using the answer from the RAG model as a reference.

Lower scores are assigned to incomplete or irrelevant answers, and higher scores indicate better relevancy.

It is calculated as the average cosine similarity between the original question and a set of artificially generated questions. These generated questions are created by reverse-engineering based on the answer.

Note: While the score usually falls between **0 and 1**, it is not guaranteed due to cosine similarity's mathematical range of **-1 to 1**.

### Formula:

![](https://hub.athina.ai/content/images/2024/11/6.png)

![](https://hub.athina.ai/content/images/2024/11/7-1.png)

Where:

*   **Ag** = embedding of the generated question 
*   **Ao** = embedding of the original question.
*   **N** = number of generated questions.

### Example:

**Questions** \= What is SpaceX?

**Answers** \= It is an American aerospace company founded in 2002.

### Solution:

To calculate the relevance of an answer to the given question, the process involves the following steps:

**Step 1:** Reverse-engineer ‘n’ variations of the question from the provided answer using a Large Language Model (LLM). This involves generating alternative samples of the original question based on the information contained in the answer.

For examples:

*   **Question 1:** “Which company was founded in 2002 and operates in aerospace?”
*   **Question 2:** “What American company in aerospace was established in 2002?”
*   **Question 3:** “Which U.S. company focused on aerospace was started in 2002?”

**Step 2:** After that answer relevancy metric is calculated using the mean cosine similarity between the generated questions and the original question.


Faithfulness
------------

The faithfulness metric evaluates whether the generated answer can be found in the provided context. The score ranges from **0** to **1**, where a higher score indicates better factual consistency.

### Formula:

![](https://hub.athina.ai/content/images/2024/11/1-3.png)

### Example:

**Questions** \= What is SpaceX and Who found it

**Answers** \= It is an American aerospace company. It was founded by Nikola Tesla.

**Contexts** \= SpaceX is an American aerospace company founded in 2002, Founded by Elon Musk, The full form of SpaceX is Space Exploration Technologies Corporation.

### Solution:

Now let’s calculate faithfulness:

First, break the generated answer into individual statements/claims.

**Statement 1:** It is an American aerospace company.

**Statement 2:** It was founded by Nikola Tesla.

After that, check each one to see if it can be supported by the given context.

**Statement 1:** Yes (supported by the context)

**Statement 1:** No (not supported by the context)

The final score for Faithfulness is:

![](https://hub.athina.ai/content/images/2024/11/2-5.png)

This shows how well the generated answers are grounded in the provided context.
