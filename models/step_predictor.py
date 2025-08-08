# models/step_predictor.py

import json
from typing import Dict, List, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


class StepPredictor:
    """Generic interface for prompting HuggingFace models to predict step n given step n-1."""
    
    def __init__(self, model_name: str, prompt_template_path: str, device: str = "auto"):
        """
        Initialize the step predictor.
        
        Args:
            model_name: HuggingFace model name (e.g., "microsoft/DialoGPT-medium")
            prompt_template_path: Path to the prompt template file
            device: Device to run the model on ("auto", "cpu", "cuda")
        """
        self.model_name = model_name
        self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set up padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" and torch.cuda.is_available() else -1
        )
        
        # Load prompt template
        with open(prompt_template_path, 'r') as f:
            self.prompt_template = f.read().strip()
    
    def format_prompt(self, data: Dict) -> str:
        """
        Format the prompt using the data from the dataset.
        
        Args:
            data: Dictionary containing 'question', 'input', 'target', etc.
            
        Returns:
            Formatted prompt string
        """
        # Extract previous steps from the input
        input_text = data.get('input', '')
        question = data.get('question', '')
        
        # Parse previous steps from input
        previous_steps = self._extract_previous_steps(input_text)
        
        return self.prompt_template.format(
            question=question,
            previous_steps=previous_steps,
            step_number=data.get('step_number', 'N')
        )
    
    def _extract_previous_steps(self, input_text: str) -> str:
        """Extract and format previous steps from input text."""
        lines = input_text.split('\n')
        previous_steps = []
        
        for line in lines:
            if line.startswith('Step '):
                previous_steps.append(line)
        
        return '\n'.join(previous_steps) if previous_steps else "No previous steps."
    
    def predict_next_step(self, data: Dict, max_length: int = 200, temperature: float = 0.7) -> str:
        """
        Predict the next step given the previous steps.
        
        Args:
            data: Dataset entry containing problem information
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            
        Returns:
            Predicted next step
        """
        prompt = self.format_prompt(data)
        
        # Generate response
        outputs = self.pipeline(
            prompt,
            max_new_tokens = max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=1
        )
        
        # Extract only the generated part (remove prompt)
        generated_text = outputs[0]['generated_text']
        response = generated_text[len(prompt):].strip()
        
        return response
    
    def batch_predict(self, data_list: List[Dict], **kwargs) -> List[str]:
        """
        Predict next steps for a batch of problems.
        
        Args:
            data_list: List of dataset entries
            **kwargs: Additional arguments for predict_next_step
            
        Returns:
            List of predicted next steps
        """
        predictions = []
        for data in data_list:
            prediction = self.predict_next_step(data, **kwargs)
            predictions.append(prediction)
        
        return predictions
    
    def evaluate_prediction(self, prediction: str, target: str) -> Dict[str, float]:
        """
        Simple evaluation metrics for predictions.
        
        Args:
            prediction: Predicted next step
            target: Ground truth next step
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Simple exact match
        exact_match = 1.0 if prediction.strip() == target.strip() else 0.0
        
        # Token overlap (basic)
        pred_tokens = set(prediction.lower().split())
        target_tokens = set(target.lower().split())
        
        if len(target_tokens) == 0:
            token_overlap = 0.0
        else:
            token_overlap = len(pred_tokens.intersection(target_tokens)) / len(target_tokens)
        
        return {
            "exact_match": exact_match,
            "token_overlap": token_overlap
        }


class DatasetLoader:
    """Helper class to load and process the step-by-step dataset."""
    
    @staticmethod
    def load_from_file(file_path: str) -> List[Dict]:
        with open(file_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def filter_by_step(data: List[Dict], step_number: Optional[int] = None) -> List[Dict]:
        """
        Filter dataset by step number.
        
        Args:
            data: List of dataset entries
            step_number: Step number to filter by (None for all)
            
        Returns:
            Filtered dataset
        """
        if step_number is None:
            return data
        
        return [entry for entry in data if entry.get('step_number') == step_number]


# Example usage function
import os

def main():
    """Run step-by-step prediction and evaluation on a dataset."""

    # Load dataset
    file_path = '/Users/rongkunzhou/Work/COT/dataset/aqua_rat_test.json'
    loader = DatasetLoader()
    data = loader.load_from_file(file_path)

    if not data:
        print("Dataset is empty or could not be loaded. Please verify the file path and content format.")
        return

    # Initialize predictor
    predictor = StepPredictor(
        model_name="microsoft/DialoGPT-medium",
        prompt_template_path="/Users/rongkunzhou/Work/COT/prompts/step_prediction_prompt.txt"
    )

    # Run predictions
    predictions = predictor.batch_predict(data)
    all_metrics = []

    dataset_name = os.path.basename(file_path)
    print("\n" + "=" * 80)
    print(f"Step Prediction Evaluation for Dataset: {dataset_name}")
    print("=" * 80)

    for idx, (example, prediction) in enumerate(zip(data, predictions), 1):
        question = example.get("question", "[Missing question]")
        previous_steps = predictor._extract_previous_steps(example.get("input", ""))
        target = example.get("target", "[Missing target]")
        metrics = predictor.evaluate_prediction(prediction, target)
        all_metrics.append(metrics)

        print(f"\nSample #{idx}")
        print(f"Question: {question}")
        print("Previous Steps:")
        print(previous_steps)
        print(f"Predicted Step: {prediction}")
        print(f"Target: {target}")
        print(f"Evaluation - Exact Match: {metrics['exact_match']}, Token Overlap: {metrics['token_overlap']:.2f}")

    # Summary metrics
    avg_exact = sum(m["exact_match"] for m in all_metrics) / len(all_metrics)
    avg_overlap = sum(m["token_overlap"] for m in all_metrics) / len(all_metrics)
    success_count = sum(m["exact_match"] for m in all_metrics)

    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print(f"Total Samples:           {len(data)}")
    print(f"Exact Match Rate:        {avg_exact:.2%}")
    print(f"Average Token Overlap:   {avg_overlap:.2%}")
    print(f"Successful Predictions:  {success_count}/{len(data)}")
    print("=" * 80)

if __name__ == "__main__":
    main()
