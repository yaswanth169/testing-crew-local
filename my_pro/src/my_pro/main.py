#!/usr/bin/env python
import sys
import warnings
from datetime import datetime
from pathlib import Path

from my_pro.crew import MyPro

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        'topic': 'AI LLMs',
        'current_year': str(datetime.now().year)
    }
    
    try:
        print(f"🚀 Starting crew with topic: {inputs['topic']}")
        print(f"📅 Current year: {inputs['current_year']}")
        print("=" * 50)
        
        # Run the crew
        crew_instance = MyPro().crew()
        result = crew_instance.kickoff(inputs=inputs)
        
        # Save the result to a file
        output_file = Path("report.md")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Crew AI Report - {inputs['topic']}\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Topic: {inputs['topic']}\n")
            f.write(f"Year: {inputs['current_year']}\n\n")
            f.write("## Results\n\n")
            f.write(str(result))
        
        print(f"\n✅ Crew completed successfully!")
        print(f"📄 Report saved to: {output_file.absolute()}")
        print(f"📊 Result: {result}")
        
    except Exception as e:
        print(f"❌ Error running crew: {e}")
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        MyPro().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        MyPro().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }
    
    try:
        MyPro().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

if __name__ == "__main__":
    run()
