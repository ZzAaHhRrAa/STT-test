from pipeline import STTPipeline

def main():
    print("🎬 Starting Speech-to-Text Evaluation Pipeline...")
    
    # Initialize the modular pipeline object once
    pipeline = STTPipeline()
    
    # Process the entire audio batch systematically
    pipeline.run_inference(csv_path="input.csv", output_path="results.csv")
    
    print("✨ Evaluation complete!")

if __name__ == "__main__":
    main()

    