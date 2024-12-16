

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
    def __init__(self, csv_path: str, max_columns: int = 20):
        self.csv_path = csv_path
        self.max_columns = max_columns  # Maximum number of columns to include in metadata
        try:
            self.df = pd.read_csv(csv_path, nrows=100000)  # Read up to 100,000 rows to optimize performance
        except UnicodeDecodeError:
            try:
                print("Failed to decode with 'utf-8', trying 'latin-1'")
                self.df = pd.read_csv(csv_path, encoding='latin-1', nrows=100000)
            except UnicodeDecodeError:
                try:
                    print("Failed to decode with 'latin-1', trying 'cp1252'")
                    self.df = pd.read_csv(csv_path, encoding='cp1252', nrows=100000)
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
        self.insights = None  # To store insights from LLM
        self.plot_reasons = None  # To store reasons for chosen plots

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
        # Convert numeric columns for correlation calculation
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix only if there are 2 or more numeric columns
        numeric_columns = list(numeric_df.columns)
        correlation_matrix = numeric_df.corr().to_dict() if len(numeric_columns) > 1 else {}
        
        # Get descriptive statistics for all columns
        stats = self.df.describe(include="all").to_dict()
        
        # Calculate missing values
        missing_values = self.df.isnull().sum().to_dict()
        
        # Limit the number of columns in metadata
        all_columns = list(self.df.columns)[:self.max_columns]
        limited_missing_values = {k: v for k, v in missing_values.items() if k in all_columns}
        limited_stats = {k: v for k, v in stats.items() if k in all_columns}
        limited_correlation = {k: v for k, v in correlation_matrix.items() if k in all_columns}

        return {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "display_columns": all_columns,
            "missing_values": limited_missing_values,
            "stats": limited_stats,
            "correlation_matrix": limited_correlation
        }

    def _choose_visualization(self) -> List[str]:
        """
        Dynamically choose the most appropriate visualization by querying the LLM
        with a concise dataset summary and store the insights and reasons.
        """
        # Possible visualization types
        visualization_types = [
            "correlation",  # Heatmap of numeric column correlations
            "boxplot",      # Numeric vs Categorical comparison
            "timeseries",   # Line plot for time-based data
            "barplot",      # Categorical distribution
            "histogram"     # Numeric distribution
        ]

        # Prepare a concise dataset summary
        dataset_summary = {
            "total_rows": self.metadata['total_rows'],
            "total_columns": self.metadata['total_columns'],
            "display_columns": self.metadata['display_columns'],
            "numeric_column_count": len([col for col in self.metadata['display_columns'] 
                                        if col in self.df.select_dtypes(include=[np.number]).columns]),
            "categorical_column_count": len([col for col in self.metadata['display_columns'] 
                                            if col not in self.df.select_dtypes(include=[np.number]).columns]),
            "missing_values": sum(self.metadata['missing_values'].values())
        }
        
        # Prepare prompt for LLM to get both visualization recommendations and insights
        prompt = f"""
        Dataset Characteristics Summary:
        {json.dumps(dataset_summary, indent=2)}

        Available Visualization Types: {visualization_types}
        Recommend 1-2 visualization types that would best represent the data's characteristics.
        Additionally, provide a very short and concise insight from the data and the reason for choosing the recommended plots.

        Output Format:
        {{
            "recommended_visualizations": ["viz_type1", "viz_type2"],
            "insights": "Concise insight from the data.",
            "reasons": "Reasons for choosing the recommended visualizations."
        }}
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
        
        # Call LLM to get visualization recommendation and insights
        result = self._call_llm(
            messages, 
            tools=[{
                "type": "function",
                "function": {
                    "name": "recommend_visualizations",
                    "description": "Recommend visualizations based on dataset characteristics and provide insights.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "recommended_visualizations": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of recommended visualization types"
                            },
                            "insights": {
                                "type": "string",
                                "description": "Concise insights derived from the data"
                            },
                            "reasons": {
                                "type": "string",
                                "description": "Reasons for choosing the recommended visualizations"
                            }
                        },
                        "required": ["recommended_visualizations", "insights", "reasons"]
                    }
                }
            }],
            tool_choice={"type": "function", "function": {"name": "recommend_visualizations"}}
        )

        # Fallback method if LLM call fails
        if not result or 'tool_calls' not in result.get('choices', [{}])[0].get('message', {}):
            print("LLM visualization recommendation failed. Using fallback method.")
            return self._choose_visualization_fallback()
        
        # Extract tool call results
        tool_call = result['choices'][0]['message']['tool_calls'][0]
        
        try:
            response_content = json.loads(tool_call['function']['arguments'])
            recommended_visualizations = response_content['recommended_visualizations']
            self.insights = response_content['insights']
            self.plot_reasons = response_content['reasons']
            print(f"Recommended Visualizations: {recommended_visualizations}")
            print(f"Insights: {self.insights}")
            print(f"Reasons: {self.plot_reasons}")
            
            # Return recommended visualizations or fallback
            return recommended_visualizations if recommended_visualizations else self._choose_visualization_fallback()
        
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Failed to parse LLM visualization recommendation: {e}. Using fallback.")
            return self._choose_visualization_fallback()

    def _choose_visualization_fallback(self) -> List[str]:
        """
        Fallback method to choose visualization if LLM recommendation fails.
        """
        # Identify column types from stats and columns
        numeric_cols = [col for col in self.metadata['display_columns'] if col in self.df.select_dtypes(include=[np.number]).columns]
        categorical_cols = [col for col in self.metadata['display_columns'] 
                             if col not in numeric_cols and 
                             (self.metadata['missing_values'].get(col, 0) < len(self.df) or 
                              len(set(self.df[col].dropna())) < 10)]
        datetime_cols = [col for col in self.metadata['display_columns'] if pd.api.types.is_datetime64_any_dtype(self.df[col])]
        
        # Priority of visualization selection
        recommended_visualizations = []
        
        if len(numeric_cols) >= 2:
            recommended_visualizations.append("correlation")
        
        if categorical_cols and numeric_cols:
            recommended_visualizations.append("boxplot")
        
        if datetime_cols and numeric_cols:
            recommended_visualizations.append("timeseries")
        
        if categorical_cols:
            recommended_visualizations.append("barplot")
        
        if numeric_cols:
            recommended_visualizations.append("histogram")
        
        return recommended_visualizations or ["histogram"]

    def plot_visualization(self):
        viz_types = self._choose_visualization()  # Assume it returns a list of visualization types
        if isinstance(viz_types, str):
            viz_types = [viz_types]  # Ensure it's always a list

        visualizations = []  # List to store generated plots
        for viz_type in viz_types:
            plt.figure(figsize=(7.11, 7.11))
            numeric_cols = [col for col in self.metadata['display_columns'] if col in self.df.select_dtypes(include=[np.number]).columns]

            if viz_type == "correlation":
                if len(numeric_cols) < 2:
                    print("Not enough numeric columns to compute correlation.")
                    continue
                corr = self.df[numeric_cols].corr()
                sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
                plt.title("Correlation Matrix")
                filename = os.path.join(self.output_dir, "correlation_heatmap.png")
            
            elif viz_type == "boxplot":
                if len(numeric_cols) < 1 or len(self.metadata['display_columns']) < 2:
                    print("Not enough columns for boxplot.")
                    continue
                numeric_col = numeric_cols[0]
                categorical_col = [col for col in self.metadata['display_columns'] if col not in numeric_cols][0]
                sns.boxplot(x=categorical_col, y=numeric_col, data=self.df)
                plt.title(f"{numeric_col} by {categorical_col}")
                filename = os.path.join(self.output_dir, "boxplot.png")
            
            elif viz_type == "timeseries":
                if len(numeric_cols) < 1:
                    print("Not enough numeric columns for timeseries.")
                    continue
                datetime_col = [col for col in self.metadata['display_columns'] if pd.api.types.is_datetime64_any_dtype(self.df[col])][0] if any(pd.api.types.is_datetime64_any_dtype(self.df[col]) for col in self.metadata['display_columns']) else None
                if not datetime_col:
                    print("No datetime column found for timeseries.")
                    continue
                numeric_col = numeric_cols[0]
                time_series_df = self.df.sort_values(by=datetime_col)
                plt.plot(time_series_df[datetime_col], time_series_df[numeric_col])
                plt.title(f"{numeric_col} Over Time")
                plt.xlabel(datetime_col)
                plt.ylabel(numeric_col)
                filename = os.path.join(self.output_dir, "timeseries.png")
            
            elif viz_type == "barplot":
                if len(self.metadata['display_columns']) < 1:
                    print("Not enough columns for barplot.")
                    continue
                categorical_col = [col for col in self.metadata['display_columns'] if col not in numeric_cols][0]
                self.df[categorical_col].value_counts().plot(kind='bar')
                plt.title(f"Distribution of {categorical_col}")
                filename = os.path.join(self.output_dir, "barplot.png")
            
            else:  # histogram
                if len(numeric_cols) < 1:
                    print("Not enough numeric columns for histogram.")
                    continue
                numeric_col = numeric_cols[0]
                plt.hist(self.df[numeric_col].dropna(), bins='auto')
                plt.title(f"Distribution of {numeric_col}")
                filename = os.path.join(self.output_dir, "histogram.png")

            plt.tight_layout()
            plt.savefig(filename, dpi=72)
            plt.close()

            visualizations.append(filename)

        self.visualization = visualizations
        return visualizations

    def generate_narrative(self):
        # Utilize stored insights and plot reasons from choose_visualization
        if not self.insights or not self.plot_reasons:
            print("Insufficient insights or reasons. Generating narrative without them.")
            metadata_analysis = "No insights available."
            visualization_insights = "No visualization insights available."
        else:
            metadata_analysis = self.insights
            visualization_insights = self.plot_reasons

        # Step 1: Generate the Markdown Story (single LLM call)
        markdown_prompt = f"""
        Dataset Analysis Compilation:

        1. Metadata Insights: {metadata_analysis}
        2. Visualization Analysis: {visualization_insights}

        Please craft a concise Markdown narrative that:
        1. Summarizes the dataset's core characteristics
        2. Highlights key insights from visualizations
        3. Provides actionable recommendations or observations
        """

        messages = [
            {"role": "system", "content": "You are a data storyteller."},
            {"role": "user", "content": markdown_prompt.strip()}
        ]

        markdown_response = self._call_llm(messages)
        
        if markdown_response and "choices" in markdown_response and len(markdown_response["choices"]) > 0:
            narrative = markdown_response["choices"][0]["message"].get("content", "").strip()
            return narrative

        return "Failed to generate the Markdown story."

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
