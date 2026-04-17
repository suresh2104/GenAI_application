from setuptools import setup, find_packages

setup(
    name="claim-automation",
    version="1.0.0",
    description="AI-powered insurance claim automation using RAG, Vision Models, and LLMs",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "streamlit>=1.28.0",
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "Pillow>=10.0.0",
        "PyMuPDF>=1.23.0",
        "opencv-python>=4.8.0",
        "requests>=2.31.0",
        "sentence-transformers>=2.2.2",
        "faiss-cpu>=1.7.4",
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
    ],
)
