import dspy
import json
import os
import random
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate

# 1. Simplified Signature
class StepPredictionSignature(dspy.Signature):
    """Predict the next step in a multi-step math problem. Keep response concise and focused."""
    question = dspy.InputField(desc="The mathematical problem statement.")
    previous_steps = dspy.InputField(desc="The steps taken so far to solve the problem.")
    next_step = dspy.OutputField(desc="The single next logical step to solve the problem. Be concise and specific.")

class MathStepPredictor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(StepPredictionSignature)

    def forward(self, question: str, previous_steps: str):
        prediction = self.predictor(question=question, previous_steps=previous_steps)
        
        # Add output length limit and cleanup
        next_step = prediction.next_step.strip()
        
        # Limit output length to prevent infinite loops
        if len(next_step) > 500:  # Set reasonable length limit
            # Truncate at first reasonable stopping point (more sentence endings)
            import re
            sentences = re.split(r'[.?!]', next_step)
            if len(sentences) > 1:
                next_step = sentences[0] + '.'
            else:
                next_step = next_step[:500] + "..."
        
        # Remove repetitive content - improved logic
        lines = next_step.split('\n')
        cleaned_lines = []
        
        # Method 1: Allow reasonable repetition (same line can appear 2-3 times)
        line_counts = {}
        for line in lines:
            line_clean = line.strip()
            if line_clean:
                line_counts[line_clean] = line_counts.get(line_clean, 0) + 1
                # Allow up to 3 occurrences of the same line
                if line_counts[line_clean] <= 3:
                    cleaned_lines.append(line)
        
        # Method 2: Detect and stop at excessive consecutive repetition
        final_lines = []
        consecutive_count = 1
        prev_line = ""
        
        for line in cleaned_lines:
            line_clean = line.strip()
            if line_clean == prev_line and line_clean:
                consecutive_count += 1
                # Stop if same line repeats more than 5 times consecutively
                if consecutive_count <= 5:
                    final_lines.append(line)
                else:
                    print(f"   Warning: Stopping due to excessive consecutive repetition")
                    break
            else:
                consecutive_count = 1
                final_lines.append(line)
                prev_line = line_clean
        
        # Method 3: Global repetition limit - if total output exceeds reasonable length due to repetition
        if len('\n'.join(final_lines)) > 1000 and len(final_lines) > 20:
            print(f"   Warning: Output too long due to repetition, truncating to first 15 lines")
            final_lines = final_lines[:15]
        
        cleaned_output = '\n'.join(final_lines[:10])  # Still limit to max 10 lines for context
        
        return dspy.Prediction(next_step=cleaned_output)

# 2. Simplified Semantic Evaluation Class
class SemanticEvaluationSignature(dspy.Signature):
    """
    Judge whether the predicted step and target step express the same meaning semantically.
    """
    question = dspy.InputField(desc="The original mathematical problem for context.")
    previous_steps = dspy.InputField(desc="The reasoning steps taken so far.")
    predicted_step = dspy.InputField(desc="The step predicted by the AI model.")
    target_step = dspy.InputField(desc="The expected/target step from the answer key.")
    
    evaluation_result = dspy.OutputField(
        desc="Answer 'Yes' if semantically equivalent, or 'No: brief reason' if different."
    )

class SemanticEvaluator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.evaluator = dspy.Predict(SemanticEvaluationSignature)

    def forward(self, question, previous_steps, predicted_step, target_step):
        # Preprocessing: clean input
        pred_clean = predicted_step.strip()
        target_clean = target_step.strip()
        
        # Pre-check: if texts are identical, return Yes directly
        if pred_clean == target_clean:
            return dspy.Prediction(evaluation_result="Yes")
        
        # Check if one contains the other (high similarity)
        if pred_clean in target_clean or target_clean in pred_clean:
            similarity_ratio = max(len(pred_clean), len(target_clean)) / min(len(pred_clean), len(target_clean))
            if similarity_ratio <= 1.2:  # Less than 20% difference
                return dspy.Prediction(evaluation_result="Yes")
        
        # Use AI evaluation
        try:
            result = self.evaluator(
                question=question,
                previous_steps=previous_steps,
                predicted_step=pred_clean,
                target_step=target_clean
            )
            
            # Fix template errors
            evaluation_result = result.evaluation_result.strip()
            if evaluation_result in ["{evaluation_result}", "evaluation_result", ""]:
                return dspy.Prediction(evaluation_result="No: Template error")
            
            return dspy.Prediction(evaluation_result=evaluation_result)
            
        except Exception as e:
            error_msg = str(e)
            
            # Handle context length errors
            if "maximum context" in error_msg.lower() or "context" in error_msg.lower():
                print(f"   Warning: Context length error, using fallback")
                return dspy.Prediction(evaluation_result="No: Context too long")
            
            # Other error fallback
            return dspy.Prediction(evaluation_result="No: Error occurred")

# 3. Helper function 
def extract_steps_and_format(input_text: str) -> str:
    lines = input_text.split("\n")
    steps = []
    
    for line in lines:
        line_stripped = line.strip()
        # More flexible step detection
        if (line_stripped.startswith("Step ") or 
            line_stripped.startswith("step ") or
            line_stripped.startswith("STEP ") or
            (len(line_stripped) > 0 and line_stripped[0].isdigit() and ":" in line_stripped) or
            line_stripped.startswith("Previous Steps:")):
            steps.append(line)
    
    if not steps: 
        return "No previous steps."
    
    # Limit number of steps to avoid overly long context
    return "\n".join(steps[:10])  # Max 10 steps

# 4. Global variables
output_file_path = None
semantic_evaluator = None
processed_count = 0
total_samples = 0  # Add global variable for total samples

# 5. Metric functions
def simple_step_match_metric(gold, pred, trace=None):
    """Simple string matching metric for compilation phase"""
    return gold.next_step.strip() == pred.next_step.strip()

def semantic_evaluate_and_save_metric(gold, pred, trace=None):
    """
    Improved semantic evaluation function with error handling and length limits
    """
    global processed_count, total_samples
    
    try:
        processed_count += 1
        print(f"Processing sample {processed_count}/{total_samples}: {gold.question[:50]}...")
        
        # Check if predicted step is too long
        if len(pred.next_step) > 1000:
            print(f"   Warning: Predicted step too long ({len(pred.next_step)} chars), truncating...")
            pred.next_step = pred.next_step[:500] + "..."
        
        # Check for obvious repetitive content - more generic approach
        lines = pred.next_step.split('\n')
        if len(lines) > 5:
            # Count repeated phrases (more than 3 occurrences)
            line_counts = {}
            for line in lines:
                clean_line = line.strip()
                if len(clean_line) > 10:  # Only check substantial lines
                    line_counts[clean_line] = line_counts.get(clean_line, 0) + 1
            
            # If any line appears more than 3 times, truncate
            max_repeats = max(line_counts.values()) if line_counts else 0
            if max_repeats > 3:
                print(f"   Warning: Detected repetitive content (max {max_repeats} repeats)")
                # Keep only first few occurrences
                seen_lines = {}
                filtered_lines = []
                for line in lines:
                    clean_line = line.strip()
                    if len(clean_line) > 10:
                        seen_lines[clean_line] = seen_lines.get(clean_line, 0) + 1
                        if seen_lines[clean_line] <= 2:  # Keep at most 2 occurrences
                            filtered_lines.append(line)
                    else:
                        filtered_lines.append(line)
                pred.next_step = '\n'.join(filtered_lines)
        
        # Check if prediction and target are identical
        is_identical = pred.next_step.strip() == gold.next_step.strip()
        
        if is_identical:
            print(f"   Success: Texts are identical")
        else:
            print(f"   Info: Texts differ, using AI evaluation")
        
        # Use semantic evaluator
        evaluation_result = semantic_evaluator(
            question=gold.question,
            previous_steps=gold.previous_steps,
            predicted_step=pred.next_step,
            target_step=gold.next_step
        )
        
        # Extract the evaluation result string from the Prediction object
        if hasattr(evaluation_result, 'evaluation_result'):
            eval_text = evaluation_result.evaluation_result.strip()
        else:
            eval_text = str(evaluation_result).strip()
        
        # Save results
        result_record = {
            "question": gold.question,
            "previous_steps": gold.previous_steps,
            "predicted_step": pred.next_step,
            "target_step": gold.next_step,
            "correct_answer": getattr(gold, 'correct_answer', ''),
            "step_number": getattr(gold, 'step_number', 1),
            "total_steps": getattr(gold, 'total_steps', 1),
            "is_multiple_choice": getattr(gold, 'is_multiple_choice', False),
            "evaluation_result": eval_text
        }
        
        with open(output_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
        
        print(f"   Success: Result: {eval_text[:60]}...")
        
        return eval_text.lower().startswith("yes")
        
    except Exception as e:
        print(f"Error processing sample {processed_count}: {e}")
        
        error_record = {
            "question": gold.question,
            "previous_steps": gold.previous_steps,
            "predicted_step": pred.next_step[:500] + "..." if len(pred.next_step) > 500 else pred.next_step,
            "target_step": gold.next_step,
            "correct_answer": getattr(gold, 'correct_answer', ''),
            "step_number": getattr(gold, 'step_number', 1),
            "total_steps": getattr(gold, 'total_steps', 1),
            "is_multiple_choice": getattr(gold, 'is_multiple_choice', False),
            "evaluation_result": f"ERROR: {str(e)[:100]}"
        }
        
        with open(output_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
        
        return simple_step_match_metric(gold, pred)

def main():
    global output_file_path, semantic_evaluator, processed_count, total_samples
    
    print("DSPy Semantic Evaluation - Complete Fixed Version")
    print("=" * 60)
    
    # Reset counter
    processed_count = 0
    
    # Configuration
    hf_token = 
    model_identifier = "huggingface/novita/meta-llama/Meta-Llama-3-8B-Instruct"
    
    try:
        # Add generation parameters to control output length
        llm = dspy.LM(
            model=model_identifier, 
            api_key=hf_token,
            max_tokens=200,  # Limit max tokens
            temperature=0.3  # Lower temperature to reduce randomness
        )
        dspy.settings.configure(lm=llm)
        print(f"Successfully connected to model: {model_identifier}")
        
        # Test model to ensure readiness (warm up)
        print("Warming up model...")
        test_evaluator = SemanticEvaluator()
        test_result = test_evaluator(
            question="Test question",
            previous_steps="Test steps",
            predicted_step="Test prediction",
            target_step="Test target"
        )
        print(f"Model warmed up successfully: {test_result}")
        
    except Exception as e:
        print(f"Model connection failed: {e}")
        return

    # Initialize semantic evaluator
    semantic_evaluator = SemanticEvaluator()
    print("Semantic evaluator initialized")

    output_dir = "/Users/rongkunzhou/Work/small-model-cot/data/result"
    os.makedirs(output_dir, exist_ok=True)

    # Setup output file
    output_file_path = os.path.join(output_dir, "semantic_evaluation_results.jsonl")
    
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    print(f"Results will be saved to: {output_file_path}")

    # Data Loading
    dataset_path = "/Users/rongkunzhou/Work/COT/testexample.json"
    if not os.path.exists(dataset_path):
        print(f"Dataset file not found: {dataset_path}")
        return
        
    try:
        with open(dataset_path, "r") as f: 
            data = json.load(f)
        print(f"Loaded {len(data)} samples")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return
    
    # Create examples with enhanced information
    all_examples = []
    
    for i, ex in enumerate(data):
        try:
            # Extract question with options if present
            if "Options:" in ex.get("input", ""):
                full_question = ex.get("input", "").split("Previous Steps:")[0].strip()
            else:
                full_question = ex["question"]
            
            # Limit question length
            if len(full_question) > 300:
                full_question = full_question[:300] + "..."
            
            # Extract additional information
            correct_answer = ex.get("correct_answer", "")
            step_number = ex.get("step_number", 1)
            total_steps = ex.get("total_steps", 1)
            is_multiple_choice = len(correct_answer) == 1 and correct_answer.isalpha()
            
            example = dspy.Example(
                question=full_question,
                previous_steps=extract_steps_and_format(ex.get("input", "")), 
                next_step=ex["target"],
                correct_answer=correct_answer,
                step_number=step_number,
                total_steps=total_steps,
                is_multiple_choice=is_multiple_choice
            ).with_inputs("question", "previous_steps")
            
            all_examples.append(example)
            print(f"Created example {i+1}: Step {step_number}/{total_steps}, Answer: {correct_answer}")
            
        except KeyError as e:
            print(f"Skipping sample {i+1} with missing key: {e}")
    
    print(f"Created {len(all_examples)} examples total")
    
    # Set global variable for total samples
    total_samples = len(all_examples)
    
    # Prepare datasets - 20% for compilation, randomly selected
    compilation_size = int(total_samples * 0.2)
    
    # Randomly select 20% for compilation
    compilation_indices = random.sample(range(total_samples), compilation_size)
    compilation_set = [all_examples[i] for i in compilation_indices]
    
    # evaluation_set maintains original order
    evaluation_set = all_examples

    print(f"Compilation set: {len(compilation_set)} samples (20% randomly selected)")
    print(f"Evaluation set: {len(evaluation_set)} samples (original order)")
    print(f"Compilation indices: {sorted(compilation_indices)}")

    # Compilation
    solver = MathStepPredictor()
    
    print("\nStarting BootstrapFewShot compilation...")
    try:
        teleprompter = BootstrapFewShot(
            metric=simple_step_match_metric, 
            max_bootstrapped_demos=4
        )
        compiled_solver = teleprompter.compile(solver, trainset=compilation_set)
        print("Compilation completed")
    except Exception as e:
        print(f"Compilation failed: {e}")
        compiled_solver = solver

    # Evaluation - process in original dataset order, maintain original order in output
    print(f"\nStarting evaluation in original dataset order...")
    print(f"Will process all {len(evaluation_set)} samples in the dataset")
    
    try:
        evaluate_instance = Evaluate(
            devset=evaluation_set, 
            metric=semantic_evaluate_and_save_metric,
            num_threads=1, 
            display_progress=True
        )
        results = evaluate_instance(compiled_solver)
        
        print(f"\nFinal Results:")
        print(f"   Processed samples: {processed_count}")
        print(f"   DSPy result: {results}")
        
        # Check final file
        with open(output_file_path, "r", encoding="utf-8") as f:
            final_lines = [line for line in f if line.strip()]
        print(f"   Final file has: {len(final_lines)} records")
        
    except Exception as e:
        print(f"Evaluation error: {e}")
    
    print("\nProgram completed!")
    print(f"Check results in: {output_file_path}")

if __name__ == "__main__":
    main()