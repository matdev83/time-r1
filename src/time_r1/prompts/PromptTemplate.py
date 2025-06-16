from typing import Any, Optional  # Added Any, Optional

import yaml
from jinja2 import (  # FileSystemLoader might be useful if templates were separate files
    Environment,
)


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
