# 🎙️ Meeting Minutes Generator

An intelligent web application that automatically transcribes meeting audio, generates summaries, and creates professional meeting minutes using AI-powered speech recognition and natural language processing.

## ✨ Features

- **🎯 Audio Transcription**: Convert speech to text using OpenAI Whisper
- **📋 Smart Summarization**: Generate concise summaries with Facebook BART
- **📄 Meeting Minutes**: Create professionally formatted meeting minutes
- **✏️ Editable Output**: Edit generated minutes before downloading
- **💾 Download Support**: Save minutes as text files with timestamps
- **🌐 Web Interface**: User-friendly Gradio web interface
- **⚡ GPU Support**: Optional GPU acceleration for faster processing

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended for large audio files)
- Internet connection (for initial model download)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MinuteMax
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv minute_max_env
   source minute_max_env/bin/activate  # On Windows: minute_max_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8080`

## 📖 Usage

1. **Enter Meeting Title** (optional)
2. **Upload Audio File** (WAV, MP3, M4A, FLAC, OGG)
3. **Click "Generate Meeting Minutes"**
4. **Review and Edit** the generated minutes
5. **Download** the final minutes as a text file

## 🎯 Supported Audio Formats

- WAV
- MP3
- M4A
- FLAC
- OGG

## 🔧 Technical Details

### Models Used
- **Transcription**: OpenAI Whisper (small) - 244M parameters
- **Summarization**: Facebook BART (large-cnn) - 400M parameters

### System Requirements
- **CPU**: Multi-core processor (Intel/AMD)
- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: 3GB+ free space for models
- **GPU**: Optional (CUDA-compatible for acceleration)

### Performance
- **Transcription Speed**: ~1-2x real-time on CPU
- **Summarization**: Near-instant for typical meeting lengths
- **Memory Usage**: ~2-3GB during processing

## 📁 Project Structure

```
MinuteMax/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── minute_max_env/    # Virtual environment
```

## 🛠️ Configuration

### Server Settings
The app runs on `localhost:8080` by default. To change:
```python
demo.launch(
    server_name="localhost",
    server_port=8080,
    share=False,
    debug=True
)
```

### Model Settings
- **Whisper Model**: `openai/whisper-small`
- **BART Model**: `facebook/bart-large-cnn`
- **Device**: Auto-detects CPU/GPU

## 🐛 Troubleshooting

### Common Issues

**Blank Page**
- Ensure you're using `localhost` not `0.0.0.0`
- Check if port 8080 is available
- Try a different browser

**Model Loading Errors**
- Check internet connection
- Ensure sufficient RAM (4GB+)
- Verify all dependencies are installed

**Audio Processing Issues**
- Use supported audio formats
- Ensure audio file is not corrupted
- Check file size (recommended <100MB)

### Getting Help

1. Check the terminal output for error messages
2. Ensure all dependencies are installed correctly
3. Verify sufficient system resources
4. Try with a smaller audio file first

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review terminal error messages
3. Open an issue on the repository

---

**Built with ❤️ using Gradio, Transformers, and PyTorch** 