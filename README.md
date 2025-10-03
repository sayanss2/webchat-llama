# 🚀 Llama.cpp + Web UI (CUDA)

## 📋 Описание
Данный проект демонстрирует запуск Llama.cpp с поддержкой CUDA и простым Web UI для взаимодействия через браузер.  
Инструкция проверена на Ubuntu 24.04.1 (WSL2).

---

## ⚙️ Шаг 1. Подготовка окружения

Устанавливаем все необходимые зависимости и CUDA Toolkit:

```bash
sudo apt-get update 
sudo apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
sudo apt install git wget curl build-essential python3 python3-pip -y 
sudo apt install cmake libopenblas-dev -y
sudo apt install python3.12-venv -y

wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin 
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget https://developer.download.nvidia.com/compute/cuda/13.0.1/local_installers/cuda-repo-wsl-ubuntu-13-0-local_13.0.1-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-13-0-local_13.0.1-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/

sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-0
sudo apt-get -y install cuda
```

После установки добавляем CUDA в переменные окружения, чтобы `nvcc` и библиотеки находились корректно:

```bash
echo 'export CUDAToolkit_ROOT=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$PATH:/usr/local/cuda/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64' >> ~/.bashrc
source ~/.bashrc
```

Проверяем наличие CUDA:
```bash
nvcc --version
nvidia-smi
```

---

## 🧱 Шаг 2. Сборка llama.cpp

Клонируем и подготавливаем проект:

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp
cp -r llama.cpp llama-web.cpp
cd llama-web.cpp
```

Устанавливаем зависимости для сборки:

```bash
sudo apt-get update
sudo apt-get install cmake build-essential libcurl4-openssl-dev -y
```

Выполняем конфигурацию с поддержкой CUDA и CURL (для HTTP API):

```bash
cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
```

Собираем основные бинарники:

```bash
cmake --build build --config Release -j --clean-first --target llama-server llama-cli llama-gguf-split
```

Проверяем результат:
```bash
ls build/bin/llama-server
```

Копируем бинарники в корень проекта:
```bash
cp build/bin/llama-* .
```

---

## 🌐 Шаг 3. Настройка Web UI

Создаем папку для веб-интерфейса:
```bash
mkdir www
cd www
```

Скопируйте в эту директорию файл `index.html` (ваш Web UI).

Возвращаемся в корень:
```bash
cd ..
```

---

## 🚀 Шаг 4. Запуск сервера

Запускаем сервер, указав путь к модели GGUF (замените путь на ваш):

```bash
./llama-server -m ~/python_ai/unsloth/gpt-oss-20b-GGUF/gpt-oss-20b-F16.gguf \
--host 0.0.0.0 \
--port 8080 \
-ngl 99 \
--threads -1 \
--ctx-size 16384 \
--embedding \
--path ./www
```

После запуска сервер будет доступен по адресу:
```
http://127.0.0.1:8080
```

---

## 💡 Примечания

- Опция `-ngl 99` позволяет загрузить модель полностью в GPU (если есть достаточно VRAM).
- Параметр `--ctx-size` можно уменьшить, если не хватает видеопамяти.
- Для теста можно использовать любую модель формата `.gguf`.

---

✅ Готово! Теперь вы можете взаимодействовать с моделью прямо из браузера через Web UI.
