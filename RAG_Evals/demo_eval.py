# First, install RAGAS
#pip install ragas

# Import necessary libraries

from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    context_relevancy,
    answer_relevancy,
    faithfulness,
    answer_correctness
)
from datasets import Dataset

# Context Precision Example
def evaluate_context_precision():
    data = [
        {
            "question": "What is SpaceX?",
            "contexts": ['SpaceX is an American aerospace company founded in 2002'],
            "ground_truth": "SpaceX is an American aerospace company"
        },
        {
            "question": "Who found it?",
            "contexts": ['SpaceX, founded by Elon Musk, is worth nearly $210 billion'],
            "ground_truth": "Founded by Elon Musk."
        },
        {
            "question": "What exactly does SpaceX do?",
            "contexts": ['The full form of SpaceX is Space Exploration Technologies Corporation'],
            "ground_truth": "SpaceX produces and operates the Falcon 9 and Falcon Heavy rockets"
        },
    ]
    
    # Convert to Dataset format
    dataset = Dataset.from_dict({
        "question": [item["question"] for item in data],
        "contexts": [item["contexts"] for item in data],
        "ground_truths": [[item["ground_truth"]] for item in data]
    })
    
    # Evaluate context precision
    result = evaluate(dataset, [context_precision])
    return result

# Context Recall Example
def evaluate_context_recall():
    data = [
        {
            "question": "What is SpaceX and Who found it?",
            "contexts": ['SpaceX is an American aerospace company founded in 2002'],
            "ground_truth": "SpaceX is an American aerospace company founded by Elon Musk"
        },
        {
            "question": "What exactly does SpaceX do?",
            "contexts": ['SpaceX produces and operates the Falcon 9 and Falcon rockets'],
            "ground_truth": "SpaceX produces and operates the Falcon 9 and Falcon Heavy rockets"
        },
    ]
    
    dataset = Dataset.from_dict({
        "question": [item["question"] for item in data],
        "contexts": [item["contexts"] for item in data],
        "ground_truths": [[item["ground_truth"]] for item in data]
    })
    
    result = evaluate(dataset, [context_recall])
    return result

# Context Relevancy Example
def evaluate_context_relevancy():
    data = [
        {
            "question": "What is SpaceX and Who found it",
            "contexts": [
                "SpaceX is an American aerospace company founded in 2002",
                "Founded by Elon Musk",
                "The full form of SpaceX is Space Exploration Technologies Corporation"
            ],
        }
    ]
    
    dataset = Dataset.from_dict({
        "question": [item["question"] for item in data],
        "contexts": [item["contexts"] for item in data]
    })
    
    result = evaluate(dataset, [context_relevancy])
    return result

# Answer Relevancy Example
def evaluate_answer_relevancy():
    data = [
        {
            "question": "What is SpaceX?",
            "contexts": ['SpaceX is an American aerospace company founded in 2002'],
            "answer": "It is an American aerospace company"
        },
        {
            "question": "Who found it?",
            "contexts": ['SpaceX, founded by Elon Musk, is worth nearly $210 billion'],
            "answer": "SpaceX founded by Elon Musk"
        }
    ]
    
    dataset = Dataset.from_dict({
        "question": [item["question"] for item in data],
        "contexts": [item["contexts"] for item in data],
        "answer": [item["answer"] for item in data]
    })
    
    result = evaluate(dataset, [answer_relevancy])
    return result

# Answer Correctness Example
def evaluate_answer_correctness():
    data = [
        {
            "question": "What is SpaceX?",
            "answer": "It is an American aerospace company",
            "ground_truth": "SpaceX is an American aerospace company"
        },
        {
            "question": "Who found it?",
            "answer": "SpaceX founded by Elon Musk",
            "ground_truth": "Founded by Elon Musk."
        }
    ]
    
    dataset = Dataset.from_dict({
        "question": [item["question"] for item in data],
        "answer": [item["answer"] for item in data],
        "ground_truths": [[item["ground_truth"]] for item in data]
    })
    
    result = evaluate(dataset, [answer_correctness])
    return result

# Faithfulness Example
def evaluate_faithfulness():
    data = [
        {
            "question": "What is SpaceX?",
            "contexts": ['SpaceX is an American aerospace company founded in 2002'],
            "answer": "It is an American aerospace company"
        },
        {
            "question": "Who found it?",
            "contexts": ['SpaceX, founded by Elon Musk, is worth nearly $210 billion'],
            "answer": "SpaceX founded by Elon Musk"
        }
    ]
    
    dataset = Dataset.from_dict({
        "question": [item["question"] for item in data],
        "contexts": [item["contexts"] for item in data],
        "answer": [item["answer"] for item in data]
    })
    
    result = evaluate(dataset, [faithfulness])
    return result

# Combined Evaluation Example
def evaluate_all_metrics():
    data = [
        {
            "question": "What is SpaceX?",
            "contexts": ['SpaceX is an American aerospace company founded in 2002'],
            "answer": "It is an American aerospace company",
            "ground_truth": "SpaceX is an American aerospace company"
        }
    ]
    
    dataset = Dataset.from_dict({
        "question": [item["question"] for item in data],
        "contexts": [item["contexts"] for item in data],
        "answer": [item["answer"] for item in data],
        "ground_truths": [[item["ground_truth"]] for item in data]
    })
    
    # Evaluate using multiple metrics
    metrics = [
        context_precision,
        context_recall,
        context_relevancy,
        answer_relevancy,
        faithfulness,
        answer_correctness
    ]
    
    result = evaluate(dataset, metrics)
    return result

# Example usage
if __name__ == "__main__":
    # Run individual evaluations
    print("Context Precision:", evaluate_context_precision())
    print("Context Recall:", evaluate_context_recall())
    print("Context Relevancy:", evaluate_context_relevancy())
    print("Answer Relevancy:", evaluate_answer_relevancy())
    print("Answer Correctness:", evaluate_answer_correctness())
    print("Faithfulness:", evaluate_faithfulness())
    
    # Run combined evaluation
    print("All Metrics:", evaluate_all_metrics())