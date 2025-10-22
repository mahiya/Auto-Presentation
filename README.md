# Auto-Presentation
以下の一連のコマンドは、リポジトリの```Auto-Presentation```フォルダに移動して、実行を行ってください。

```sh
git clone https://github.com/mahiya/Auto-Presentation
cd Auto-Presentation
```

## 開発端末でのアプリケーションの実行

### Python 仮想環境の構築
```sh
# Python 仮想環境の作成
python -m venv venv_demo
.\venv_demo\Scripts\activate

# 必要なパッケージのインストール
pip install -r requirements.txt
```

### .env ファイルの作成
.env ファイルを以下の内容で作成します。
```sh
AZURE_OPENAI_ENDPOINT="https://[Azure OpenAI Service のアカウント名].openai.azure.com/"
AZURE_OPENAI_API_KEY="[Azure OpenAI Service のキー]"
AZURE_OPENAI_API_VERSION="2025-01-01-preview"
AZURE_GPT5_DEPLOYMENT="gpt-5"
AZURE_TTS_DEPLOYMENT="gpt-4o-mini-tts"
```

### AI モデルのデプロイ
使用する Azure AI Foundry / Azure OpenAI Service アカウントにて以下のモデルをデプロイします。
- gpt-5
- gpt-4o-mini-tts

### アプリケーションの実行
以下のコマンドを実行して、アプリケーションを実行します。
```sh
streamlit run app.py
```

## アプリケーションの Azure へのデプロイ (ローカル端末から)

### Azure CLI のインストール
以下のサイトを参考に Azure CLI をインストールします。  
[Azure CLI をインストールする方法 | Microsoft Learn](https://learn.microsoft.com/ja-jp/cli/azure/install-azure-cli?view=azure-cli-latest)

### Azure CLI の設定
以下のコマンドを実行して、Microsoft Entra テナントへログインして、使用する Azure サブスクリプションを設定します。

```sh
# Microsoft Entra テナントへログイン
az login -t [ログインするMicrosoftEntraテナントのID]

# 使用する Azure サブスクリプションを設定
az account set -s [使用するするAzureサブスクリプションのID]

# 使用する Azure サブスクリプションの確認
az account show
```

### アプリケーションの Azure へのデプロイ
以下のコマンドで、デプロイする Web アプリケーションの環境変数を設定します (.env と同じ内容)

```sh
AZURE_OPENAI_ENDPOINT="https://[Azure OpenAI Service のアカウント名].openai.azure.com/"
AZURE_OPENAI_API_KEY="[Azure OpenAI Service のキー]"
AZURE_OPENAI_API_VERSION="2025-01-01-preview"
AZURE_GPT5_DEPLOYMENT="gpt-5"
AZURE_TTS_DEPLOYMENT="gpt-4o-mini-tts"
```

以下のコマンドで、アプリケーションを Azure Web Apps へデプロイします。
```sh
chmod +x deploy.sh
./deploy.sh
```

## アプリケーションの Azure へのデプロイ (Azure CLI から)
```sh
# リポジトリの取得
git clone https://github.com/mahiya/Auto-Presentation
cd Auto-Presentation

# 環境変数の設定
AZURE_OPENAI_ENDPOINT="https://[Azure OpenAI Service のアカウント名].openai.azure.com/"
AZURE_OPENAI_API_KEY="[Azure OpenAI Service のキー]"
AZURE_OPENAI_API_VERSION="2025-01-01-preview"
AZURE_GPT5_DEPLOYMENT="gpt-5"
AZURE_TTS_DEPLOYMENT="gpt-4o-mini-tts"

# デプロイ
chmod +x deploy.sh
./deploy.sh
```