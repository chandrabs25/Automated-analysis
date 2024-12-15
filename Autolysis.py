

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests",
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   
# ]
# ///
import os
import sys
import json
import requests
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

class AutolysisAnalyzer:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        try:
            self.df = pd.read_csv(csv_path)  # Try default 'utf-8' first
        except UnicodeDecodeError:
            try:
                print("Failed to decode with 'utf-8', trying 'latin-1'")
                self.df = pd.read_csv(csv_path, encoding='latin-1')
            except UnicodeDecodeError:
                try:
                    print("Failed to decode with 'latin-1', trying 'cp1252'")
                    self.df = pd.read_csv(csv_path, encoding='cp1252')
                except UnicodeDecodeError:
                    print("Failed to decode with common encodings. Please check the file's encoding.")
                    raise  # Re-raise the exception if none of the encodings work

        # Extract base name for folder creation
        self.base_name = os.path.splitext(os.path.basename(csv_path))[0]
        self.output_dir = self.base_name
        os.makedirs(self.output_dir, exist_ok=True)  # Create the folder

        # Check for token
        self.api_token = os.environ.get("AIPROXY_TOKEN")
        if not self.api_token:
            raise ValueError("AIPROXY_TOKEN environment variable not set.")

        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        self.api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        self.metadata = self.analyze_data_structure()
        self.visualization = None

    def _log_request_cost(self, response):
        response_json = response.json()  # Parse the JSON response
        print(f"Request Cost: ${response_json.get('cost', 'Unknown')}")
        print(f"Monthly Cost: ${response_json.get('monthlyCost', 'Unknown')}")
        print(f"Monthly Requests: {response_json.get('monthlyRequests', 'Unknown')}")

    def _call_llm(self, messages, tools=None, tool_choice="auto"):
        payload = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.7
        }
        if tools is not None:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        response = requests.post(self.api_url, headers=self.headers, json=payload)
        if response.status_code != 200:
            print("LLM call failed:", response.text)
            return None

        self._log_request_cost(response)
        return response.json()

    def analyze_data_structure(self) -> Dict[str, Any]:
        numeric_columns = list(self.df.select_dtypes(include=[np.number]).columns)
        categorical_columns = list(self.df.select_dtypes(include=['object']).columns)
        datetime_columns = list(self.df.select_dtypes(include=['datetime64']).columns)
        
        return {
            "total_rows": len(self.df),
            "all_columns": list(self.df.columns),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "datetime_columns": datetime_columns
        }

    def _choose_visualization(self) -> str:
        """
        Dynamically choose the most appropriate visualization by querying the LLM
        with simplified descriptive statistics about the dataset.
        """
        # Simplified descriptive stats (only count and column names)
        desc_stats = {}

        # Numeric columns descriptive stats (only count and column names)
        if self.metadata['numeric_columns']:
            desc_stats['numeric_columns'] = {
                'count': len(self.metadata['numeric_columns']),
                'column_names': self.metadata['numeric_columns']
            }

        # Categorical columns descriptive stats (only count and column names)
        if self.metadata['categorical_columns']:
            desc_stats['categorical_columns'] = {
                'count': len(self.metadata['categorical_columns']),
                'column_names': self.metadata['categorical_columns']
            }

        # Datetime columns descriptive stats (only count and column names)
        if self.metadata['datetime_columns']:
            desc_stats['datetime_columns'] = {
                'count': len(self.metadata['datetime_columns']),
                'column_names': self.metadata['datetime_columns']
            }

        # List of possible visualization types
        visualization_types = [
            "correlation",  # Heatmap of numeric column correlations
            "boxplot",      # Numeric vs Categorical comparison
            "timeseries",   # Line plot for time-based data
            "barplot",      # Categorical distribution
            "histogram"     # Numeric distribution
        ]

        # Prepare prompt for LLM
        prompt = f"""
        Given the following dataset descriptive statistics, recommend 1-2 visualization types that would best represent the data's characteristics.

        Dataset Description:
        {json.dumps(desc_stats, indent=2)}

        Available Visualization Types: {visualization_types}

        Output Format:
        Recommended Visualizations: [viz_type1, viz_type2]
        Reasoning: Brief explanation of why these visualizations are recommended
        """
        
        messages = [
            {
                "role": "system", 
                "content": "You are a data visualization expert helping to choose the most informative plots for a given dataset."
            },
            {
                "role": "user",
                "content": prompt.strip()
            }
        ]
        
        # Call LLM to get visualization recommendation
        result = self._call_llm(
            messages, 
            tools=[{
                "type": "function",
                "function": {
                    "name": "recommend_visualizations",
                    "description": "Recommend visualizations based on dataset characteristics",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "recommended_visualizations": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of recommended visualization types"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Brief explanation for visualization choices"
                            }
                        },
                        "required": ["recommended_visualizations", "reasoning"]
                    }
                }
            }],
            tool_choice={"type": "function", "function": {"name": "recommend_visualizations"}}
        )

        # Default fallback
        if not result or 'tool_calls' not in result.get('choices', [{}])[0].get('message', {}):
            print("LLM visualization recommendation failed. Using default visualization.")
            return self._choose_visualization_fallback()

        # Extract tool call results
        tool_call = result['choices'][0]['message']['tool_calls'][0]
        
        try:
            recommended_visualizations = json.loads(tool_call['function']['arguments'])['recommended_visualizations']
            print(recommended_visualizations)
            reasoning = json.loads(tool_call['function']['arguments'])['reasoning']
            
            # Log the reasoning
            print(f"Visualization Recommendation Reasoning: {reasoning}")
            
            # Return first recommended visualization or fallback
            return recommended_visualizations if recommended_visualizations else self._choose_visualization_fallback()
        
        except (json.JSONDecodeError, KeyError, IndexError):
            print("Failed to parse LLM visualization recommendation. Using fallback.")
            return self._choose_visualization_fallback()


    def _choose_visualization_fallback(self) -> str:
        """
        Fallback method to choose visualization if LLM recommendation fails.
        Mimics original logic with some minor improvements.
        """
        numeric_cols = self.metadata['numeric_columns']
        categorical_cols = self.metadata['categorical_columns']
        datetime_cols = self.metadata['datetime_columns']

        # Priority of visualization selection
        if len(numeric_cols) >= 2:
            return "correlation"
        
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            return "boxplot"
        
        if len(datetime_cols) > 0 and len(numeric_cols) > 0:
            return "timeseries"
        
        if len(categorical_cols) > 0:
            return "barplot"
        
        return "histogram"

    def plot_visualization(self):
        viz_types = self._choose_visualization()  # Assume it returns a list of visualization types
        if isinstance(viz_types, str):
            viz_types = [viz_types]  # Ensure it's always a list

        visualizations = []  # List to store generated plots
        for viz_type in viz_types:
            plt.figure(figsize=(7.11, 7.11))
            numeric_cols = self.metadata['numeric_columns']

            if viz_type == "correlation":
                if len(numeric_cols) < 2:
                    print("Not enough numeric columns to compute correlation.")
                    continue
                corr = self.df[numeric_cols].corr()
                sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
                plt.title("Correlation Matrix")
                filename = os.path.join(self.output_dir, "correlation_heatmap.png")
            
            elif viz_type == "boxplot":
                numeric_col = self.metadata['numeric_columns'][0]
                categorical_col = self.metadata['categorical_columns'][0]
                sns.boxplot(x=categorical_col, y=numeric_col, data=self.df)
                plt.title(f"{numeric_col} by {categorical_col}")
                filename = os.path.join(self.output_dir, "boxplot.png")
            
            elif viz_type == "timeseries":
                datetime_col = self.metadata['datetime_columns'][0]
                numeric_col = self.metadata['numeric_columns'][0]
                time_series_df = self.df.sort_values(by=datetime_col)
                plt.plot(time_series_df[datetime_col], time_series_df[numeric_col])
                plt.title(f"{numeric_col} Over Time")
                plt.xlabel(datetime_col)
                plt.ylabel(numeric_col)
                filename = os.path.join(self.output_dir, "timeseries.png")
            
            elif viz_type == "barplot":
                categorical_col = self.metadata['categorical_columns'][0]
                self.df[categorical_col].value_counts().plot(kind='bar')
                plt.title(f"Distribution of {categorical_col}")
                filename = os.path.join(self.output_dir, "barplot.png")
            
            else:  # histogram
                numeric_col = self.metadata['numeric_columns'][0]
                plt.hist(self.df[numeric_col], bins='auto')
                plt.title(f"Distribution of {numeric_col}")
                filename = os.path.join(self.output_dir, "histogram.png")

            plt.tight_layout()
            plt.savefig(filename, dpi=72)
            plt.close()

            visualizations.append(filename)

        self.visualization = visualizations
        return visualizations


    def generate_narrative(self):
        total_rows = self.metadata['total_rows']
        cols = self.metadata['all_columns']
        numeric_count = len(self.metadata['numeric_columns'])
        categorical_count = len(self.metadata['categorical_columns'])
        datetime_count = len(self.metadata['datetime_columns'])

        prompt = f"""
    We analyzed the dataset {os.path.basename(self.csv_path)}:

    - Total observations: {total_rows}
    - Example columns: {', '.join(cols[:5])}...
    - Found {numeric_count} numeric columns
    - Found {categorical_count} categorical columns
    - Found {datetime_count} datetime columns

    We generated visualizations based on the dataset's characteristics.
    The visualizations are included below.

    Please write a Markdown story about the data analysis:
    1. Briefly describe the dataset
    2. Describe the visualizations and their insights
    3. Suggest what can be done with these insights
    """
        
        messages = [
            {
                "role": "system", 
                "content": "You are a data storyteller."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt.strip()
                    }
                ]
            }
        ]

        # Only add image if visualization was created
        if self.visualization and isinstance(self.visualization, list):
            for image_path in self.visualization:
                with open(image_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    messages[1]["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_string}",
                                "detail": "low"
                            }
                        }
                    )


        result = self._call_llm(messages)
        if result and "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"].get("content", "").strip()
        return "No narrative generated."

    def run_analysis(self):
        # Generate visualizations
        visualization_paths = self.plot_visualization()

        # Generate narrative (README)
        narrative = self.generate_narrative()

        # Save the README in the output directory
        readme_filename = os.path.join(self.output_dir, "README.md")
        with open(readme_filename, "w") as f:
            f.write("# Autolysis Data Analysis Report\n\n")
            
            if visualization_paths:
                for viz_path in visualization_paths:
                    # Adjust image path for each visualization and include in the README
                    f.write(f"![Data Visualization]({os.path.basename(viz_path)})\n\n")

            f.write(narrative)


def main():
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    analyzer = AutolysisAnalyzer(csv_path)
    analyzer.run_analysis()
    print(f"Analysis complete. Please review {analyzer.output_dir}/README.md and visualization files.")

if __name__ == "__main__":
    main()