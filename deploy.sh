#!/bin/bash -e

# デプロイ設定
REGION='japaneast'
RESOURCE_GROUP="aoai-demo-apps"
APP_PLAN_NAME="aoai-demo-apps-plan"
APP_NAME="auto-presentation-"$(date +%s)
SKU='B3'

# アプリケーションをデプロイする
az webapp up \
    --resource-group $RESOURCE_GROUP \
    --name $APP_NAME \
    --plan $APP_PLAN_NAME \
    --sku $SKU \
    --location $REGION \
    --runtime 'PYTHON:3.11'

# Azure Web Apps の環境変数
az webapp config appsettings set \
    --resource-group $RESOURCE_GROUP \
    --name $APP_NAME \
    --settings "SCM_DO_BUILD_DURING_DEPLOYMENT"="true" \
               "AZURE_OPENAI_ENDPOINT=$AZURE_OPENAI_ENDPOINT" \
               "AZURE_OPENAI_API_KEY=$AZURE_OPENAI_API_KEY" \
               "AZURE_OPENAI_API_VERSION=$AZURE_OPENAI_API_VERSION" \
               "AZURE_GPT5_DEPLOYMENT=$AZURE_GPT5_DEPLOYMENT" \
               "AZURE_TTS_DEPLOYMENT=$AZURE_TTS_DEPLOYMENT"

# スタートアップコマンドを設定する
az webapp config set \
    --resource-group $RESOURCE_GROUP \
    --name $APP_NAME \
    --startup-file "python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0"
