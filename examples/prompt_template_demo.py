import os

from time_r1.prompts.PromptTemplate import PromptTemplate


def demo() -> None:
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
    dummy_template_path = "examples/dummy_prompt_templates.yaml"
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

        task_prompt = prompt_builder.render_template(
            "task_definition", task_name="Inventory Check", forecast_horizon=3
        )
        print("\n--- Rendered Single Template (Task Definition) ---")
        print(task_prompt)
    finally:
        os.remove(dummy_template_path)


if __name__ == "__main__":
    demo()
