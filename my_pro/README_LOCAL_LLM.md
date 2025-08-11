# Local LLaMA Integration with Crew AI

This project demonstrates how to integrate a local LLaMA model with Crew AI, allowing your agents to use a locally hosted model instead of external API services.

## 🚀 Features

- **Local Model Integration**: Use your local LLaMA 3.2 1B Instruct model
- **Crew AI Compatible**: Full integration with Crew AI's agent system
- **Configurable**: Easy configuration through YAML files
- **GPU/CPU Support**: Automatic device detection and optimization
- **Error Handling**: Robust error handling and logging

## 📁 Project Structure

```
my_pro/
├── src/my_pro/
│   ├── config/
│   │   ├── agents.yaml          # Agent configurations with LLM settings
│   │   ├── tasks.yaml           # Task definitions
│   │   ├── llms.yaml            # LLM configuration
│   │   └── __init__.py          # Configuration loader
│   ├── tools/
│   │   ├── local_llm.py         # Local LLaMA LLM integration
│   │   └── __init__.py
│   ├── crew.py                  # Updated crew with local LLM
│   └── main.py
├── requirements.txt              # Python dependencies
├── test_local_llm.py            # Test script
└── README_LOCAL_LLM.md          # This file
```

## 🛠️ Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download LLaMA Model**:
   - Download the LLaMA 3.2 1B Instruct model to your local system
   - Update the model path in `src/my_pro/config/llms.yaml`

3. **Verify Model Path**:
   Ensure your model path exists and contains the necessary model files:
   ```
   C:/devhome/projects/models/meta-llama-Llama-3-2-1B-Instruct/
   ├── config.json
   ├── pytorch_model.bin
   ├── tokenizer.model
   └── tokenizer_config.json
   ```

## ⚙️ Configuration

### LLM Configuration (`llms.yaml`)

```yaml
local_llama:
  name: "Local LLaMA 3.2 1B Instruct"
  type: "local"
  model_path: "C:/devhome/projects/models/meta-llama-Llama-3-2-1B-Instruct/meta-llama-Llama-3-2-1B-Instruct"
  max_tokens: 512
  temperature: 0.7
  top_p: 0.9
  repetition_penalty: 1.1
  device: "auto"  # Will automatically use CUDA if available, otherwise CPU
```

### Agent Configuration (`agents.yaml`)

```yaml
researcher:
  role: "{topic} Senior Data Researcher"
  goal: "Uncover cutting-edge developments in {topic}"
  backstory: "You're a seasoned researcher..."
  llm: local_llama          # Specify which LLM to use
  verbose: true
  allow_delegation: false
```

## 🔧 Usage

### 1. Basic Integration

The local LLM is automatically integrated into your crew:

```python
from my_pro.crew import MyPro

# Create crew instance
crew = MyPro()

# Run your crew - agents will automatically use the local LLM
result = crew.kickoff()
```

### 2. Testing the Integration

Run the test script to verify everything works:

```bash
python test_local_llm.py
```

### 3. Custom Model Path

You can override the model path when creating the LLM:

```python
from my_pro.tools.local_llm import LocalLLaMALLM

# Custom model path
custom_llm = LocalLLaMALLM(
    model_path="/path/to/your/model"
)
```

## 📊 Model Information

Get detailed information about your loaded model:

```python
from my_pro.tools.local_llm import local_llm

# Check if model is loaded
print(local_llm.is_model_loaded())

# Get model information
info = local_llm.get_model_info()
print(info)
```

## 🚨 Troubleshooting

### Common Issues

1. **Model Path Not Found**:
   - Verify the model path in `llms.yaml`
   - Ensure the directory contains all required model files

2. **CUDA Out of Memory**:
   - Reduce `max_tokens` in the configuration
   - Use CPU mode by setting `device: "cpu"`

3. **Import Errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python path and module structure

4. **Model Loading Failures**:
   - Verify model files are not corrupted
   - Check available disk space and memory

### Debug Mode

Enable detailed logging by modifying the logging level in `local_llm.py`:

```python
logging.basicConfig(level=logging.DEBUG)
```

## 🔄 Updating Configuration

### Adding New LLMs

1. Add configuration to `llms.yaml`:
   ```yaml
   new_model:
     name: "New Model Name"
     type: "local"
     model_path: "/path/to/model"
     # ... other parameters
   ```

2. Update agent configurations to use the new LLM:
   ```yaml
   agent_name:
     # ... other config
     llm: new_model
   ```

### Modifying Model Parameters

Update the parameters in `llms.yaml` and restart your application:

```yaml
local_llama:
  max_tokens: 1024        # Increase token limit
  temperature: 0.5        # Reduce randomness
  top_p: 0.95            # Increase diversity
```

## 📈 Performance Tips

1. **GPU Usage**: Ensure CUDA is available for optimal performance
2. **Memory Management**: Monitor memory usage and adjust `max_tokens` accordingly
3. **Batch Processing**: Consider processing multiple requests in batches
4. **Model Optimization**: Use model quantization for reduced memory usage

## 🤝 Contributing

To extend this integration:

1. **Add New Model Types**: Extend the `LocalLLaMALLM` class
2. **Improve Error Handling**: Add more specific error types and recovery
3. **Add Metrics**: Implement performance monitoring and logging
4. **Optimize Generation**: Improve response quality and speed

## 📚 Additional Resources

- [Crew AI Documentation](https://docs.crewai.com/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [LLaMA Model Information](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)

## 📄 License

This project follows the same license as your main project. Ensure compliance with LLaMA model licensing requirements.
