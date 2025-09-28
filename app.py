import gradio as gr
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor
)
from transformers.pipelines import pipeline
import librosa
import numpy as np
from datetime import datetime
import re
import os
import traceback

class MeetingMinutesGenerator:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print(f"Using device: {self.device}")
        print(f"Using dtype: {self.torch_dtype}")
        
        # Initialize models with error handling
        self.transcription_pipe = None
        self.summarization_pipe = None
        
        self.setup_models()
    
    def setup_models(self):
        """Setup both models with comprehensive error handling"""
        try:
            self.setup_transcription_model()
        except Exception as e:
            print(f"Failed to load transcription model: {e}")
            traceback.print_exc()
        
        try:
            self.setup_summarization_model()
        except Exception as e:
            print(f"Failed to load summarization model: {e}")
            traceback.print_exc()
    
    def setup_transcription_model(self):
        """Setup Whisper model for audio transcription"""
        model_id = "openai/whisper-small"
        
        print(f"Loading transcription model: {model_id}")
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=self.torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        model.to(self.device)
        
        processor = AutoProcessor.from_pretrained(model_id)
        
        self.transcription_pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16 if self.device.startswith("cuda") else 1,  # Reduce batch size for CPU
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        print("✓ Transcription model loaded successfully")
    
    def setup_summarization_model(self):
        """Setup summarization model"""
        print("Loading summarization model: facebook/bart-large-cnn")
        
        self.summarization_pipe = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn",
            device=self.device if self.device.startswith("cuda") else -1  # Use CPU if not CUDA
        )
        print("✓ Summarization model loaded successfully")
    
    def transcribe_audio(self, audio_path, progress=None):
        """Transcribe audio file to text"""
        if self.transcription_pipe is None:
            return "Error: Transcription model not loaded. Please check the console for error details."
        
        try:
            if progress:
                progress(0.1, desc="Loading audio file...")
            
            # Verify file exists
            if not os.path.exists(audio_path):
                return f"Error: Audio file not found at {audio_path}"
            
            # Load and preprocess audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            if progress:
                progress(0.3, desc="Transcribing audio...")
            
            # Transcribe
            result = self.transcription_pipe(audio)
            print(f"DEBUG: Transcription result type: {type(result)}")
            
            # Handle different result formats
            if isinstance(result, dict):
                transcript = result.get("text", "")
            elif isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict):
                    transcript = result[0].get("text", "")
                else:
                    transcript = str(result[0])
            else:
                transcript = str(result)
            
            if progress:
                progress(0.6, desc="Processing transcript...")
            
            if not transcript.strip():
                return "Warning: No speech detected in the audio file."
            
            return transcript.strip()
            
        except Exception as e:
            print(f"Transcription error: {e}")
            traceback.print_exc()
            return f"Error during transcription: {str(e)}"
    
    def summarize_text(self, text, max_length=150, min_length=50):
        """Summarize the transcribed text"""
        if self.summarization_pipe is None:
            return "Error: Summarization model not loaded. Please check the console for error details."
        
        if not text or len(text.strip()) < 50:
            return "Text too short to summarize effectively."
        
        try:
            # Split long text into chunks if needed
            max_chunk_length = 1024
            if len(text) > max_chunk_length:
                chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
                summaries = []
                
                for chunk in chunks:
                    if len(chunk.strip()) > 50:  # Only summarize substantial chunks
                        try:
                            summary = self.summarization_pipe(
                                chunk, 
                                max_length=min(max_length//len(chunks), 142),  # BART max is 142
                                min_length=max(min_length//len(chunks), 10), 
                                do_sample=False
                            )
                            summaries.append(summary[0]['summary_text'])
                        except Exception as chunk_error:
                            print(f"Error summarizing chunk: {chunk_error}")
                            continue
                
                if not summaries:
                    return "Error: Unable to summarize any text chunks."
                
                # Combine summaries
                combined_summary = " ".join(summaries)
                
                # Re-summarize if too long
                if len(combined_summary) > max_length * 2:
                    try:
                        final_summary = self.summarization_pipe(
                            combined_summary, 
                            max_length=min(max_length, 142), 
                            min_length=min(min_length, 10), 
                            do_sample=False
                        )
                        return final_summary[0]['summary_text']
                    except:
                        return combined_summary
                else:
                    return combined_summary
            else:
                summary = self.summarization_pipe(
                    text, 
                    max_length=min(max_length, 142),  # BART limit
                    min_length=min(min_length, 10), 
                    do_sample=False
                )
                return summary[0]['summary_text']
                
        except Exception as e:
            print(f"Summarization error: {e}")
            traceback.print_exc()
            return f"Error during summarization: {str(e)}"
    
    def extract_key_points(self, text):
        """Extract key points from the text"""
        if not text:
            return []
            
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # Simple keyword-based extraction
        keywords = ['decision', 'action', 'task', 'deadline', 'responsible', 'next', 'follow', 'agreed']
        key_points = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                key_points.append(sentence)
        
        return key_points[:5]  # Return top 5 key points
    
    def format_meeting_minutes(self, transcript, summary, meeting_title="Meeting"):
        """Format the final meeting minutes"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        key_points = self.extract_key_points(transcript)
        
        # Estimate duration based on word count (average speaking rate: 150 words/minute)
        word_count = len(transcript.split()) if transcript else 0
        estimated_duration = max(word_count // 150, 1)
        
        minutes = f"""# Meeting Minutes: {meeting_title}

**Date & Time:** {current_time}
**Duration:** Approximately {estimated_duration} minutes (estimated)

## Executive Summary
{summary}

## Key Discussion Points
"""
        
        if key_points:
            for i, point in enumerate(key_points, 1):
                minutes += f"{i}. {point}\n"
        else:
            minutes += "• Please refer to the full transcript for detailed discussion points\n"
        
        minutes += f"""
## Action Items
• [To be filled based on meeting discussion]
• [Add specific tasks and deadlines]
• [Assign responsible parties]

## Next Steps
• [Schedule follow-up meeting if needed]
• [Review action items progress]

---
**Full Transcript:**
{transcript}
"""
        
        return minutes
    
    def process_meeting_audio(self, audio_file, meeting_title, progress=gr.Progress()):
        """Main function to process audio and generate meeting minutes"""
        if audio_file is None:
            return "Please upload an audio file", "", ""
        
        try:
            progress(0.0, desc="Starting transcription...")
            
            # Transcribe audio
            transcript = self.transcribe_audio(audio_file, progress)
            if transcript.startswith("Error") or transcript.startswith("Warning"):
                return transcript, "", ""
            
            progress(0.7, desc="Generating summary...")
            
            # Summarize transcript
            summary = self.summarize_text(transcript)
            if summary.startswith("Error"):
                return transcript, summary, ""
            
            progress(0.9, desc="Formatting meeting minutes...")
            
            # Generate meeting minutes
            meeting_minutes = self.format_meeting_minutes(transcript, summary, meeting_title or "Meeting")
            
            progress(1.0, desc="Complete!")
            
            return transcript, summary, meeting_minutes
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return error_msg, "", ""

def create_interface():
    """Create and return the Gradio interface"""
    # Initialize the generator
    print("Initializing Meeting Minutes Generator...")
    generator = MeetingMinutesGenerator()
    
    # Custom CSS for dark theme and better styling
    custom_css = """
    :root {
        --primary: #3b82f6;
        --primary-dark: #2563eb;
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --bg-card: #1e293b;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --border-color: #334155;
        --success: #10b981;
        --error: #ef4444;
    }
    
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }
    
    .header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 1.5rem 0;
        background: linear-gradient(90deg, #1e40af 0%, #1e3a8a 100%);
        border-radius: 0 0 12px 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .header h1 {
        margin-bottom: 0.5rem;
        color: white !important;
        font-weight: 700;
        font-size: 2rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .header p {
        color: #e2e8f0 !important;
        margin-top: 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .card {
        background: var(--bg-card) !important;
        border-radius: 10px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        margin-bottom: 1.5rem !important;
        border: 1px solid var(--border-color) !important;
    }
    
    .card h2, .card h3 {
        color: var(--text-primary) !important;
        font-size: 1.25rem !important;
        margin-top: 0 !important;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 0.75rem;
        margin-bottom: 1rem !important;
    }
    
    .btn-primary {
        background: var(--primary) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
    }
    
    .btn-primary:hover {
        background: var(--primary-dark) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    }
    
    .btn-secondary {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        padding: 0.5rem 1.25rem !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    
    .btn-secondary:hover {
        background: #2d3748 !important;
        transform: translateY(-1px);
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
        margin-left: 0.5rem;
    }
    
    .status-success {
        background: rgba(16, 185, 129, 0.1) !important;
        color: var(--success) !important;
        border: 1px solid rgba(16, 185, 129, 0.2) !important;
    }
    
    .status-error {
        background: rgba(239, 68, 68, 0.1) !important;
        color: var(--error) !important;
        border: 1px solid rgba(239, 68, 68, 0.2) !important;
    }
    
    /* Input fields */
    .gradio-textbox, .gradio-textarea {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
    }
    
    .gradio-textbox:focus, .gradio-textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
    }
    
    /* Tabs */
    .gradio-tab-nav {
        border-bottom: 1px solid var(--border-color) !important;
        margin-bottom: 1rem !important;
    }
    
    .gradio-tab-nav button {
        color: var(--text-secondary) !important;
        padding: 0.75rem 1.25rem !important;
        border: none !important;
        background: transparent !important;
        border-bottom: 2px solid transparent !important;
        transition: all 0.2s ease !important;
    }
    
    .gradio-tab-nav button.selected {
        color: var(--primary) !important;
        border-bottom: 2px solid var(--primary) !important;
    }
    
    .gradio-tab-nav button:hover {
        color: var(--text-primary) !important;
        background: rgba(59, 130, 246, 0.1) !important;
    }
    
    /* Accordion */
    .gradio-accordion {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        margin-top: 1.5rem !important;
    }
    
    .gradio-accordion-header {
        color: var(--text-primary) !important;
        padding: 1rem 1.5rem !important;
        font-weight: 500 !important;
    }
    
    .gradio-accordion-content {
        padding: 1rem 1.5rem !important;
        border-top: 1px solid var(--border-color) !important;
    }
    
    /* Progress bar */
    .progress-bar {
        background: var(--primary) !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4b5563;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #6b7280;
    }
    """
    
    with gr.Blocks(title="Meeting Minutes Generator", css=custom_css) as demo:
        # Header Section
        with gr.Row():
            with gr.Column(scale=12):
                with gr.Column(elem_classes="header"):
                    gr.Markdown("""
                    # 🎙️ Meeting Minutes Generator
                    Automatically transcribe, summarize, and format meeting minutes from audio recordings
                    """)
        
        # Main Content
        with gr.Row():
            # Left Panel - Inputs
            with gr.Column(scale=1):
                with gr.Group(elem_classes="card"):
                    gr.Markdown("### 📋 Meeting Details")
                    meeting_title = gr.Textbox(
                        label="Meeting Title",
                        placeholder="E.g., Weekly Team Sync",
                        value="Team Meeting",
                        lines=1
                    )
                    
                    audio_input = gr.Audio(
                        label="Upload Meeting Audio",
                        type="filepath",
                        interactive=True
                    )
                    
                    with gr.Row():
                        process_btn = gr.Button(
                            "🚀 Generate Minutes",
                            variant="primary",
                            size="lg",
                            elem_classes="btn-primary"
                        )
                        clear_btn = gr.Button("🔄 Clear", variant="secondary")
                
                # System Status Card
                with gr.Group(elem_classes="card"):
                    gr.Markdown("### ⚙️ System Status")
                    
                    model_status = gr.HTML(
                        f"""
                        <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                            <div>Transcription: <span class="status-badge status-{"success" if generator.transcription_pipe else "error"}">
                                {'✅ Loaded' if generator.transcription_pipe else '❌ Failed'}
                            </span></div>
                            <div>Summarization: <span class="status-badge status-{"success" if generator.summarization_pipe else "error"}">
                                {'✅ Loaded' if generator.summarization_pipe else '❌ Failed'}
                            </span></div>
                            <div>Device: <strong>{'GPU' if 'cuda' in str(generator.device) else 'CPU'}</strong></div>
                        </div>
                        """
                    )
            
            # Right Panel - Outputs
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("📝 Transcript"):
                        transcript_output = gr.Textbox(
                            label="",
                            lines=15,
                            max_lines=20,
                            show_copy_button=True,
                            container=False
                        )
                    
                    with gr.TabItem("📋 Summary"):
                        summary_output = gr.Textbox(
                            label="",
                            lines=8,
                            max_lines=15,
                            show_copy_button=True,
                            container=False
                        )
                    
                    with gr.TabItem("📄 Meeting Minutes"):
                        meeting_minutes_output = gr.Textbox(
                            label="",
                            lines=15,
                            max_lines=30,
                            show_copy_button=True,
                            interactive=True,
                            container=False
                        )
                
                # Action Buttons
                with gr.Row():
                    download_btn = gr.Button(
                        "💾 Download Minutes",
                        variant="primary",
                        elem_classes="btn-secondary"
                    )
                    clear_output_btn = gr.Button("🗑️ Clear Output", variant="secondary")
                
                download_file = gr.File(label="Download Link", visible=False)
        
        # Footer with Tips
        with gr.Row():
            with gr.Accordion("💡 Tips for Best Results", open=False):
                gr.Markdown("""
                - 🎤 Use clear audio with minimal background noise
                - 🎙️ Ensure speakers are close to the microphone
                - ⏱️ For long meetings (>30 min), consider splitting into segments
                - 📝 Review and edit the generated minutes as needed
                - 🔄 If models fail to load, check your internet connection
                """)
        
        # Event Handlers
        def clear_inputs():
            return [None, "Team Meeting", "", "", ""]
        
        def clear_outputs():
            return ["", "", ""]
        
        # Button Actions
        process_btn.click(
            fn=generator.process_meeting_audio,
            inputs=[audio_input, meeting_title],
            outputs=[transcript_output, summary_output, meeting_minutes_output],
            show_progress="minimal"
        )
        
        clear_btn.click(
            fn=clear_inputs,
            outputs=[audio_input, meeting_title, transcript_output, summary_output, meeting_minutes_output]
        )
        
        clear_output_btn.click(
            fn=clear_outputs,
            outputs=[transcript_output, summary_output, meeting_minutes_output]
        )
        
        download_btn.click(
            fn=download_minutes,
            inputs=meeting_minutes_output,
            outputs=download_file
        )
    
    return demo

def download_minutes(minutes_text):
    if not minutes_text:
        return None
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"meeting_minutes_{timestamp}.txt"
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(minutes_text)
        return file_path
    except Exception as e:
        print(f"Download error: {e}")
        return None

if __name__ == "__main__":
    try:
        print("Starting Meeting Minutes Generator...")
        demo = create_interface()
        
        print("Launching Gradio interface...")
        demo.launch(
            server_name="localhost",
            server_port=8080,
            share=False,
            debug=True,
            show_error=True
        )
    except Exception as e:
        print(f"Failed to start server: {e}")
        traceback.print_exc()