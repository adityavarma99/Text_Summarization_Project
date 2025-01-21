from textSummarizer.components.data_transformation import DataTransformation
from textSummarizer.entity import DataTransformationConfig

config = DataTransformationConfig(
    data_path="E:/Data science/Text_summarization/artifacts/data_transformation/samsum_dataset",
    root_dir="E:/Data science/Text_summarization/artifacts/data_transformation",
    tokenizer_name="google/pegasus-xsum"
)

dt = DataTransformation(config=config)
print(f"Available methods: {dir(dt)}")
