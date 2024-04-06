```
!CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.2.42 --force-reinstall --upgrade --no-cache-dir --verbose

pip install \
    huggingface_hub \
    flask \
    pyyaml
```