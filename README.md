## Table of Contents
- [Explainability Module](#explainability-module)
- [Setup](#setup)
  - [Download Source Code](#download-source-code)
  - [Navigate to the Project Directory](#navigate-to-the-project-directory)
  - [Create Conda Environment](#create-conda-environment)
  - [Install Environment Dependencies](#install-environment-dependencies)
  - [Resolve Conflicts](#resolve-conflicts)
- [gRPC Service Documentation](#grpc-service-documentation)
  - [Overview](#overview)
  - [Service Definition](#service-definition)
  - [RPC Methods](#rpc-methods)
    - [GetExplanation](#1-getexplanation)
    - [ApplyAffectedActions](#2-applyaffectedactions)
    - [GetFeatureImportance](#3-getfeatureimportance)
  - [Data Structures](#data-structures)
  - [Example Usage (Python)](#example-usage-python)
  - [Conclusion](#conclusion)

# Explainability Module
The Explainability Module component is used to enhance the trustworthiness of the model. <be>
It extends the original explainability algorithms to provide explanations for the experimental process and the hyperparameters of the trained ML model. 

Right now we have implemented on our module the below Explainability Algorithms:

For Pipeline Explanations:
- PDPlots
- ALEPlots
- Local Counterfactual Explanations
  
For Model Explanations:
- PDPlots
- ALEPlots
- Local Counterfactual Explanations
- Global Counterfactual Explanations
- Prototypes

# Setup

### Download Source Code

```shell
git clone https://github.com/ExperimentLens/explainability-module.git
```
### Navigate to the Project Directory
Change your current directory to the project directory

### Create Conda Enviroment
```shell

# using Conda
conda create -n xxp python=3.10.14
```

### Install enviroment dependencies
```shell
pip install -r requirements.txt
```
### Resolve conflicts
```shell
pip install xport==3.6.1
pip install pandas==1.5.3
```


# gRPC Service Documentation: Explainability Module

## Overview
The `Explanations` gRPC service provides model and pipeline explainability and feature importance insights. It supports the following RPC methods:

- `GetExplanation`: Generates explanations for a given model and dataset.
- `ApplyAffectedActions`: Applies identified actions to affected clusters for the global counterfactual method.
- `GetFeatureImportance`: Computes feature importance for a model.

## Service Definition
```proto
service Explanations {
    rpc GetExplanation (ExplanationsRequest) returns (ExplanationsResponse);
    rpc ApplyAffectedActions (ApplyAffectedActionsRequest) returns (ApplyAffectedActionsResponse);
    rpc GetFeatureImportance (FeatureImportanceRequest) returns (FeatureImportanceResponse);
}
```

---

## RPC Methods

### 1. GetExplanation
#### Description
Retrieves explanations for a model based on given parameters.

#### Request: `ExplanationsRequest`
| Field | Type | Description |
|--------|------|-------------|
| `explanation_type` | `string` | Type of explanation (e.g., featureExplanation, hyperparameterExplanation). |
| `explanation_method` | `string` | Explanation method (e.g., pdp, ale, counterfactuals, global_counterfactuals). |
| `model` | `repeated string` | Paths to model files. |
| `data` | `string` | Path to the dataset file. |
| `train_index` | `repeated int32` | List of dataset indexes for training. |
| `test_index` | `repeated int32` | List of dataset indexes for testing. |
| `target_column` | `string` | Target variable column name. |
| `hyper_configs` | `map<string, hyperparameters>` | Dictionary of hyperparameter configurations and respeted metric value. |
| `query` | `string` | Query for local counterfactual explanation and prototypes. |
| `gcf_size` | `int32` | Size of Global Counterfactual (GCF). |
| `cf_generator` | `string` | Counterfactual generator method for global counterfactual explanations. |
| `cluster_action_choice_algo` | `string` | Algorithm for selecting cluster actions for global counterfactual explanations. |

#### Response: `ExplanationsResponse`
| Field | Type | Description |
|--------|------|-------------|
| `explainability_type` | `string` | Type of explainability used. |
| `explanation_method` | `string` | Explanation method applied. |
| `explainability_model` | `string` | Name of the stored model. |
| `plot_name` | `string` | Name of the generated plot. |
| `plot_descr` | `string` | Description of the plot. |
| `features` | `Features` | Features used in the explanation. |
| `xAxis`, `yAxis`, `zAxis` | `Axis` | Axes of the plot. |
| `table_contents` | `map<string, TableContents>` | Explanation table details. |
| `TotalEffectiveness` | `float` | Overall effectiveness of actions for global counterfactual method. |
| `TotalCost` | `float` | Total cost of actions for global counterfactual method. |

---

### 2. ApplyAffectedActions
#### Description
Applies affected actions to clusters, used in global counterfactuals method.

#### Request: `ApplyAffectedActionsRequest`
*(No fields required)*

#### Response: `ApplyAffectedActionsResponse`
| Field | Type | Description |
|--------|------|-------------|
| `applied_affected_actions` | `map<string, TableContents>` | Table of applied actions for global counterfactual method. |

---

### 3. GetFeatureImportance
#### Description
Computes feature importance for a given dataset and model.

#### Request: `FeatureImportanceRequest`
| Field | Type | Description |
|--------|------|-------------|
| `data` | `string` | Path to dataset file. |
| `target_column` | `string` | Target variable in the dataset. |
| `test_index` | `repeated int32` | List of dataset indexes for testing. |
| `model` | `repeated string` | Paths to model files. |

#### Response: `FeatureImportanceResponse`
| Field | Type | Description |
|--------|------|-------------|
| `feature_importances` | `repeated FeatureImportance` | List of top 5 important features. |

---

## Data Structures

### `hyperparameters`
Defines hyperparameter configurations.
```proto
message hyperparameters {
    message HyperparameterList {
        string values = 1;
        string type = 2;
    }
    map<string, HyperparameterList> hyperparameter = 1;
    float metric_value = 2;
}
```


### `Axis`
Represents an axis in a visualization.
```proto
message Axis {
    string axis_name = 1;
    repeated string axis_values = 2;
    string axis_type = 3;
}
```

### `TableContents`
Defines tabular data.
```proto
message TableContents {
    int32 index = 1;
    repeated string values = 2;
    repeated string colour = 3;
}
```

### `EffCost`
Represents effectiveness and cost of actions.
```proto
message EffCost {
    double eff = 1;
    double cost = 2;
}
```

### `FeatureImportance`
Stores feature importance scores.
```proto
message FeatureImportance {
    string feature_name = 1;
    double importance_score = 2;
}
```

---

## Example Usage (Python)
```python
import grpc
from my_service_pb2 import ExplanationsRequest
from my_service_pb2_grpc import ExplanationsStub

channel = grpc.insecure_channel("localhost:50051")
stub = ExplanationsStub(channel)

request = ExplanationsRequest(
    explanation_type="featureExplanation",
    explanation_method="pdp",
    model=["/path/to/model"],
    data="/path/to/data.csv",
    train_index=[1,2,3,4..]
    target_column="Outcome"
)

response = stub.GetExplanation(request)
print(response)
```

---

## Conclusion
This documentation provides an overview of the `Explanations` gRPC service, including request/response structures, field descriptions, and example usage. Let me know if you need modifications or additional details!


