import json
import os
from dotenv import load_dotenv
from create_prompt import create_constraint_prompt, create_reward_prompt
import google.generativeai as genai
from openai import OpenAI

# Add Groq import, handle gracefully if missing
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Add Anthropic import, handle gracefully if missing
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Load environment variables
load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", None)

# Normalize provider, default to gemini if not set
provider = (LLM_PROVIDER or "gemini").lower()

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

class ConstraintGenerator:
    def __init__(self):
        if provider == "gemini":
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        elif provider == "openai":
            self.model = OpenAI(api_key=OPENAI_API_KEY)
        elif provider == "groq":
            if not GROQ_AVAILABLE:
                raise ImportError("groq package is not installed. Please install it (`pip install groq`).")
            self.model = Groq(api_key=GROQ_API_KEY)
        elif provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package is not installed. Please install it (`pip install anthropic`).")
            self.model = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        else:
            raise ValueError(f"Invalid LLM provider: {provider}")
        self.max_constraints = 3

    def generate_constraints(self, instructions_json):
        prompt = create_constraint_prompt(instructions_json["room_type"], instructions_json["query"], self.max_constraints)
        # print("[SAUGAT] Prompt: ", prompt)
        response_text = None
        if provider == "gemini":
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
        elif provider == "openai":
            response = self.model.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.choices[0].message.content.strip()
        elif provider == "groq":
            response = self.model.chat.completions.create(
                model="moonshotai/kimi-k2-instruct-0905",
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.choices[0].message.content.strip()
        elif provider == "anthropic":
            # Uses the Claude chat API
            response = self.model.messages.create(
                model="claude-sonnet-4-5",  # or use "claude-3-sonnet-20240229" etc.
                max_tokens=2048,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            # anthropic returns content as a list of content blocks; join for plain text
            response_text = ""
            if hasattr(response, "content") and isinstance(response.content, list):
                for blk in response.content:
                    if isinstance(blk, dict) and "text" in blk:
                        response_text += blk["text"]
                    elif isinstance(blk, str):
                        response_text += blk
            elif hasattr(response, "content") and isinstance(response.content, str):
                response_text = response.content
            else:
                response_text = str(response)
            response_text = response_text.strip()
        # At this point, response_text is expected to be JSON as specified in the prompt
        try:
            constraints_json = json.loads(response_text)
        except Exception as e:
            # If not valid JSON, try extracting JSON substring
            import re
            json_match = re.search(r"\{[\s\S]*\}", response_text)
            if json_match:
                constraints_json = json.loads(json_match.group(0))
            else:
                raise RuntimeError(f"Failed to parse JSON from response: {response_text}") from e
        return constraints_json

class RewardGenerator:
    def __init__(self):
        if provider == "gemini":
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        elif provider == "openai":
            self.model = OpenAI(api_key=OPENAI_API_KEY)
        elif provider == "groq":
            if not GROQ_AVAILABLE:
                raise ImportError("groq package is not installed. Please install it (`pip install groq`).")
            self.model = Groq(api_key=GROQ_API_KEY)
        elif provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package is not installed. Please install it (`pip install anthropic`).")
            self.model = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        else:
            raise ValueError(f"Invalid LLM provider: {provider}")

    def generate_reward(self, instructions_json):
        prompt = create_reward_prompt(instructions_json["room_type"], instructions_json["query"], instructions_json["constraint"])
        # print("[SAUGAT] Prompt: ", prompt)
        response_text = None
        if provider == "gemini":
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
        elif provider == "openai":
            response = self.model.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.choices[0].message.content.strip()
        elif provider == "groq":
            response = self.model.chat.completions.create(
                model="moonshotai/kimi-k2-instruct-0905",
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.choices[0].message.content.strip()
        elif provider == "anthropic":
            response = self.model.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=2048,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            response_text = ""
            if hasattr(response, "content") and isinstance(response.content, list):
                for blk in response.content:
                    if isinstance(blk, dict) and "text" in blk:
                        response_text += blk["text"]
                    elif isinstance(blk, str):
                        response_text += blk
            elif hasattr(response, "content") and isinstance(response.content, str):
                response_text = response.content
            else:
                response_text = str(response)
            response_text = response_text.strip()
        if response_text.startswith("```python"):
            response_text = response_text[9:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        return {
            "success": True,
            "raw_response": response_text
        }

if __name__ == "__main__":
    print("=" * 50)
    print("\n")
    instructions_json = {
        "room_type": "bedroom",
        "query": "kids room"
    }

    constraint_generator = ConstraintGenerator()
    reward_generator = RewardGenerator()
    reward_code_dir = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/dynamic_constraint_rewards/dynamic_reward_functions"
    os.makedirs(reward_code_dir, exist_ok=True)

    constraints = constraint_generator.generate_constraints(instructions_json)
    print(constraints)

    # For each constraint, call the llm to generate a runnable reward function.
    for i in range(len(constraints["constraints"])):
        constraint = constraints["constraints"][i]
        # print(f"[SAUGAT] Constraint {i+1}: {constraint}")
        result = reward_generator.generate_reward({"room_type": instructions_json["room_type"], "query": instructions_json["query"], "constraint": constraint})
        code = result["raw_response"]
        # print(f"[SAUGAT] Reward code {i+1}: {code}")

        # Write the code to a file
        with open(f"{reward_code_dir}/code_{i}.py", "w") as f:
            f.write(code)
        # print(f"[SAUGAT] Reward code written to {reward_code_dir}/code_{i}.py")