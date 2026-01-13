# FBI UCR Crime Analysis System - Architecture

## System Overview

```mermaid
flowchart TB
    subgraph Users["Users"]
        analyst["Policy Analysts<br/>Law Enforcement<br/>Researchers"]
    end

    subgraph AI["AI Layer"]
        agent["AI Agent<br/>(LibreChat)"]
    end

    subgraph OpenShift["OpenShift Cluster"]
        subgraph MCP["MCP Server"]
            tools["Crime Analysis Tools<br/>• Forecasting<br/>• Historical Trends<br/>• Comparisons"]
        end

        subgraph Prediction["Prediction API"]
            models["Time-Series Models<br/>• Prophet<br/>• ARIMA"]
        end
    end

    subgraph External["External Data"]
        fbi["FBI Crime Data<br/>Explorer API"]
    end

    analyst -->|"Natural Language<br/>Questions"| agent
    agent -->|"MCP Protocol"| tools
    tools -->|"REST API"| models
    tools -->|"Historical Data"| fbi
    models -->|"Training Data"| fbi
    agent -->|"Insights &<br/>Forecasts"| analyst
```

## Executive Summary

The FBI UCR Crime Analysis System transforms complex FBI crime statistics into actionable intelligence through AI-powered natural language interactions. Users ask questions in plain English about crime trends and forecasts. An AI agent interprets these questions, retrieves data from the FBI's official Crime Data Explorer API, and leverages pre-trained machine learning models to generate predictions with confidence intervals.

## Key Components

| Component | Purpose |
|-----------|---------|
| Prediction API | Serves ML models for crime forecasting |
| MCP Server | Exposes crime analysis tools for AI agents |
| FBI CDE API | Official data source for crime statistics |
| OpenShift | Secure, scalable cloud platform |
