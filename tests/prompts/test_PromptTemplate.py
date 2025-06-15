from pathlib import Path

import pytest
import yaml

# Assuming PYTHONPATH is set up for 'src' or pytest handles it.
# If 'src' is the root for 'time_r1', then this import should work.
from time_r1.prompts.PromptTemplate import PromptTemplate

# Define the content of the temporary YAML file
TEST_TEMPLATES_CONTENT = {
    "task_definition": "Test Task: {{ task_name }}",
    "dataset_description": "Test Dataset: {{ dataset_name }}",
    "channel_information": "Channels: {% for ch in channels %}{{ ch }}{% if not loop.last %}, {% endif %}{% endfor %}",
    "testing_data": "Data: {{ data_points | join(', ') }}",
    "format_instruction": "<think>Think about {{thought_process}}.</think><answer>Answer: {{final_answer}}</answer>",
}


@pytest.fixture
def temp_template_file(tmp_path: Path) -> Path:
    """Creates a temporary YAML template file for testing."""
    file_path = tmp_path / "test_templates.yaml"
    with open(file_path, "w") as f:
        yaml.dump(TEST_TEMPLATES_CONTENT, f)
    return file_path


@pytest.fixture
def prompt_template_instance(temp_template_file: Path) -> PromptTemplate:
    """Provides a PromptTemplate instance initialized with the temporary template file."""
    return PromptTemplate(str(temp_template_file))


def test_initialization(prompt_template_instance: PromptTemplate):
    """Tests successful initialization of PromptTemplate."""
    assert prompt_template_instance is not None
    assert "task_definition" in prompt_template_instance.templates
    assert (
        prompt_template_instance.templates["task_definition"]
        == TEST_TEMPLATES_CONTENT["task_definition"]
    )


def test_initialization_file_not_found():
    """Tests that FileNotFoundError is raised for a non-existent template file."""
    with pytest.raises(FileNotFoundError):
        PromptTemplate("non_existent_template_file.yaml")


def test_render_task_definition(prompt_template_instance: PromptTemplate):
    """Tests rendering of the 'task_definition' template."""
    context = {"task_name": "Sample Task"}
    expected_output = "Test Task: Sample Task"
    rendered = prompt_template_instance.render_template("task_definition", **context)
    assert rendered == expected_output


def test_render_dataset_description(prompt_template_instance: PromptTemplate):
    """Tests rendering of the 'dataset_description' template."""
    context = {"dataset_name": "Sample Dataset"}
    expected_output = "Test Dataset: Sample Dataset"
    rendered = prompt_template_instance.render_template(
        "dataset_description", **context
    )
    assert rendered == expected_output


def test_render_channel_information(prompt_template_instance: PromptTemplate):
    """Tests rendering of the 'channel_information' template."""
    context = {"channels": ["Ch1", "Ch2", "Ch3"]}
    expected_output = "Channels: Ch1, Ch2, Ch3"
    rendered = prompt_template_instance.render_template(
        "channel_information", **context
    )
    assert rendered == expected_output


def test_render_testing_data(prompt_template_instance: PromptTemplate):
    """Tests rendering of the 'testing_data' template."""
    context = {"data_points": [10, 20, 30]}
    expected_output = "Data: 10, 20, 30"
    rendered = prompt_template_instance.render_template("testing_data", **context)
    assert rendered == expected_output


def test_render_format_instruction(prompt_template_instance: PromptTemplate):
    """Tests rendering of the 'format_instruction' template."""
    context = {"thought_process": "some logic", "final_answer": "expected result"}
    expected_output = (
        "<think>Think about some logic.</think><answer>Answer: expected result</answer>"
    )
    rendered = prompt_template_instance.render_template("format_instruction", **context)
    assert rendered == expected_output


def test_render_template_not_found(prompt_template_instance: PromptTemplate):
    """Tests that ValueError is raised if a template name is not found."""
    with pytest.raises(ValueError) as excinfo:
        prompt_template_instance.render_template("non_existent_template", foo="bar")
    assert "Template 'non_existent_template' not found." in str(excinfo.value)


def test_render_full_prompt_snapshot_and_format(
    prompt_template_instance: PromptTemplate,
):
    """
    Tests rendering of the full prompt and validates its structure and content.
    Uses a snapshot approach for the combined output.
    """
    task_ctx = {"task_name": "Forecasting"}
    dataset_ctx = {"dataset_name": "Sales"}
    channel_ctx = {"channels": ["A", "B"]}
    testing_data_ctx = {"data_points": [1, 2, 3]}
    format_instr_ctx = {"thought_process": "complex logic", "final_answer": "42"}

    full_prompt = prompt_template_instance.render_full_prompt(
        task_context=task_ctx,
        dataset_context=dataset_ctx,
        channel_context=channel_ctx,
        testing_data_context=testing_data_ctx,
        format_instruction_context=format_instr_ctx,
    )

    # Expected parts based on TEST_TEMPLATES_CONTENT and above contexts
    expected_task = "Test Task: Forecasting"
    expected_dataset = "Test Dataset: Sales"
    expected_channels = "Channels: A, B"
    expected_data = "Data: 1, 2, 3"
    expected_format = (
        "<think>Think about complex logic.</think><answer>Answer: 42</answer>"
    )

    # The PromptTemplate.render_full_prompt joins parts with "\n\n"
    expected_snapshot = (
        f"{expected_task}\n\n"
        f"{expected_dataset}\n\n"
        f"{expected_channels}\n\n"
        f"{expected_data}\n\n"
        f"{expected_format}"
    )

    assert (
        full_prompt == expected_snapshot
    ), "Full prompt does not match the expected snapshot."
    assert (
        "<think>" in full_prompt and "</think>" in full_prompt
    ), "Missing <think></think> tags."
    assert (
        "<answer>" in full_prompt and "</answer>" in full_prompt
    ), "Missing <answer></answer> tags."


def test_render_full_prompt_with_empty_format_instruction_context(
    prompt_template_instance: PromptTemplate,
):
    """
    Tests render_full_prompt when format_instruction_context is None (should default to {}).
    This assumes the 'format_instruction' template can handle an empty context
    or has defaults for its variables.
    """
    task_ctx = {"task_name": "Forecasting"}
    dataset_ctx = {"dataset_name": "Sales"}
    channel_ctx = {"channels": ["A", "B"]}
    testing_data_ctx = {"data_points": [1, 2, 3]}
    # format_instr_ctx is deliberately None or {}
    # TEST_TEMPLATES_CONTENT["format_instruction"] requires 'thought_process' and 'final_answer'
    # So, passing an empty context will cause a Jinja2 UndefinedError if not handled.
    # The current PromptTemplate.py implementation passes {} if None is given.
    # The Jinja2 template itself will fail if variables are not provided and have no defaults.

    # Let's test the scenario where the template expects variables, but they are not provided.
    # The default Jinja2 environment renders undefined variables as empty strings.
    expected_empty_format_rendered = (
        "<think>Think about .</think><answer>Answer: </answer>"
    )

    rendered_format_instruction = prompt_template_instance.render_template(
        "format_instruction", **{}  # Empty context
    )
    assert rendered_format_instruction == expected_empty_format_rendered

    # Now test the full prompt
    full_prompt = prompt_template_instance.render_full_prompt(
        task_context=task_ctx,
        dataset_context=dataset_ctx,
        channel_context=channel_ctx,
        testing_data_context=testing_data_ctx,
        format_instruction_context={},  # Explicitly empty
    )

    expected_task = prompt_template_instance.render_template(
        "task_definition", **task_ctx
    )
    expected_dataset = prompt_template_instance.render_template(
        "dataset_description", **dataset_ctx
    )
    expected_channels = prompt_template_instance.render_template(
        "channel_information", **channel_ctx
    )
    expected_data = prompt_template_instance.render_template(
        "testing_data", **testing_data_ctx
    )
    # expected_format is the one derived from empty context

    expected_snapshot_with_empty_format = (
        f"{expected_task}\n\n"
        f"{expected_dataset}\n\n"
        f"{expected_channels}\n\n"
        f"{expected_data}\n\n"
        f"{expected_empty_format_rendered}"
    )
    assert full_prompt == expected_snapshot_with_empty_format
