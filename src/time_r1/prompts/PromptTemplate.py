from typing import Any, Optional  # Added Any, Optional

import yaml
from jinja2 import (
    Environment,
)  # FileSystemLoader might be useful if templates were separate files


class PromptTemplate:
    def __init__(self, template_file_path: str):
        """
        Initializes the PromptTemplate by loading templates from a YAML file.

        Args:
            template_file_path (str): Path to the YAML file containing Jinja2 templates.
        """
        with open(template_file_path, "r") as f:
            self.templates = yaml.safe_load(f)

        # Initialize Jinja2 environment
        # Using a basic environment; for more complex scenarios like loading templates from a directory,
        # one might use FileSystemLoader(searchpath="directory_of_templates")
        self.jinja_env = Environment(
            loader=None
        )  # No loader needed as templates are strings

    def render_template(self, template_name: str, **context) -> str:
        """
        Renders a specific template by its name using the provided context.

        Args:
            template_name (str): The name of the template to render (e.g., 'task_definition').
            **context: Keyword arguments representing the context for rendering the template.

        Returns:
            str: The rendered template string.

        Raises:
            ValueError: If the template_name is not found in the loaded templates.
        """
        template_string = self.templates.get(template_name)
        if template_string is None:
            raise ValueError(f"Template '{template_name}' not found.")

        jinja_template = self.jinja_env.from_string(template_string)
        return jinja_template.render(**context)

    def render_full_prompt(
        self,
        task_context: dict,
        dataset_context: dict,
        channel_context: dict,
        testing_data_context: dict,
        format_instruction_context: Optional[dict[Any, Any]] = None,
    ) -> str:
        """
        Renders the full prompt by combining all standard template components.

        Args:
            task_context (dict): Context for the 'task_definition' template.
            dataset_context (dict): Context for the 'dataset_description' template.
            channel_context (dict): Context for the 'channel_information' template.
            testing_data_context (dict): Context for the 'testing_data' template.
            format_instruction_context (Optional[dict[Any, Any]], optional): Context for the 'format_instruction' template.
                                                                          Defaults to an empty dict if None.

        Returns:
            str: The complete, rendered prompt string.
        """
        # Ensure context is a dict for rendering, even if None was passed.
        effective_format_instruction_context: dict[Any, Any] = (
            format_instruction_context if format_instruction_context is not None else {}
        )

        parts = []
        parts.append(self.render_template("task_definition", **task_context))
        parts.append(self.render_template("dataset_description", **dataset_context))
        parts.append(self.render_template("channel_information", **channel_context))
        parts.append(self.render_template("testing_data", **testing_data_context))
        parts.append(
            self.render_template(
                "format_instruction", **effective_format_instruction_context
            )
        )

        return "\n\n".join(parts)


if __name__ == "__main__":
    # This is example usage code, not part of the class itself.
    # It assumes 'prompt_templates.yaml' is in 'templates/' relative to this script if run directly.
    # For the actual project structure, the path should be accurate.

    # Create a dummy template file for testing
    dummy_template_content = """
task_definition: |
  Analyze {{ task_name }} for {{ forecast_horizon }} steps.
dataset_description: |
  Dataset: {{ dataset_name }}
channel_information: |
  Channels:
  {% for ch in channels %}
  - {{ ch.name }}
  {% endfor %}
testing_data: |
  Timestamps: {{ timestamps | join(', ') }}
format_instruction: |
  Format: {{ format_type }}
"""
    dummy_template_path = "src/time_r1/prompts/templates/dummy_prompt_templates.yaml"
    with open(dummy_template_path, "w") as f:
        f.write(dummy_template_content)

    try:
        prompt_builder = PromptTemplate(template_file_path=dummy_template_path)

        full_prompt = prompt_builder.render_full_prompt(
            task_context={"task_name": "Sales Forecasting", "forecast_horizon": 7},
            dataset_context={"dataset_name": "WeeklySales"},
            channel_context={
                "channels": [
                    {"name": "sales", "description": "Units sold"},
                    {"name": "price", "description": "Price per unit"},
                ]
            },
            testing_data_context={
                "timestamps": ["2023-01-01", "2023-01-08"],
                "historical_series": {"sales": [100, 110]},
            },
            format_instruction_context={
                "format_type": "list of values",
                "forecast_horizon": 7,
            },
        )
        print("--- Rendered Full Prompt ---")
        print(full_prompt)

        # Test rendering a single template
        task_prompt = prompt_builder.render_template(
            "task_definition", task_name="Inventory Check", forecast_horizon=3
        )
        print("\n--- Rendered Single Template (Task Definition) ---")
        print(task_prompt)

    except Exception as e:
        print(f"An error occurred during example usage: {e}")
    finally:
        # Clean up the dummy file
        import os

        os.remove(dummy_template_path)
        # Attempt to remove the directory if it's empty, otherwise ignore
        try:
            os.rmdir(
                "src/time_r1/prompts/templates/"
            )  # This will fail if .keep is there
        except OSError:
            pass  # Directory not empty or does not exist, which is fine for cleanup
