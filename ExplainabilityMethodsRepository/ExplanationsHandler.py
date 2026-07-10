import ast
import logging

import dice_ml
import numpy as np
import pandas as pd
import torch
from aix360.algorithms.protodash import ProtodashExplainer
from captum.attr import IntegratedGradients, Saliency
from raiutils.exceptions import UserConfigValidationException
from sklearn.inspection import partial_dependence, permutation_importance

import xai_service_pb2
from ExplainabilityMethodsRepository.ALE_generic import ale
from ExplainabilityMethodsRepository.config import shared_resources
from ExplainabilityMethodsRepository.pdp import (partial_dependence_1D,
                                                 partial_dependence_2D)
from ExplainabilityMethodsRepository.src.glance.iterative_merges.iterative_merges import (
    C_GLANCE, cumulative)
from ExplainabilityMethodsRepository.segmentation import (
    treat_input,
    model_wrapper, model_wrapper_roi,
    compute_csi_rmse,
    replacement_feature_importance,
    parse_instance_from_request,
    df_to_instances,
    compress_attributions,
    spatiotemporal_to_filtered_long_df,
    spatial_multichannel_to_filtered_long_df,
)
from modules.lib import *
from modules.lib import (_load_dataset, _load_model,
                         _load_multidimensional_array)

logger = logging.getLogger(__name__)

class BaseExplanationHandler:
    """Base class for all explanation handlers."""
    
    def handle(self, request, explanation_type):
        raise NotImplementedError("Subclasses should implement this method")
    
    def _load_dataset(self,data_path):
        pass

    def _load_or_train_surrogate_model(self, hyperparameters, metrics):
        """Helper to load or train surrogate model (same as before)."""
  
        logger.info("Surrogate model does not exist. Training a new one.")
        surrogate_model = proxy_model(hyperparameters, metrics, 'XGBoostRegressor')
        # joblib.dump(surrogate_model, models[model_name]['pdp_ale_surrogate_model'])
        return surrogate_model
        
    def _load_or_train_cf_surrogate_model(self, hyper_configs,query):
        surrogate_model , proxy_dataset = instance_proxy(hyper_configs, query)
        return surrogate_model, proxy_dataset


class GLANCEHandler(BaseExplanationHandler):
    def handle(self, request, explanation_type):
        if explanation_type == 'featureExplanation':
            gcf_size = request.gcf_size  # Global counterfactual size
            cf_generator = request.cf_generator  # Counterfactual generator method
            cluster_action_choice_algo = request.cluster_action_choice_algo

            model_path = request.model
            train_data = _load_dataset(request.data.X_train)
            train_labels = _load_dataset(request.data.Y_train)
            train_data['label'] = train_labels

            test_data = _load_dataset(request.data.X_test)
            test_labels = _load_dataset(request.data.Y_test)
            y_pred = _load_dataset(request.data.Y_pred)
            test_data['label'] = test_labels

            dataset = pd.concat([train_data, test_data], ignore_index=True)

            model, name = _load_model(model_path[0])

            # preds = model.predict(test_data)
            test_data['target'] = y_pred['predictions']
            affected = test_data[test_data.target == 0]
            affected = affected.drop(columns=['label'])            
            shared_resources["affected"] = affected.drop(columns=['target'])

            global_method = C_GLANCE(
                model=model,
                initial_clusters=50,
                final_clusters=gcf_size,
                num_local_counterfactuals=10,
            )

            global_method.fit(
                dataset.drop(columns=['label']),
                dataset['label'],
                test_data.drop(columns=['label']),
                test_data.drop(columns=['label','target']).columns.tolist(),
                cf_generator=cf_generator,
                cluster_action_choice_algo=cluster_action_choice_algo
            )
            try:
                logger.info("Generating global counterfactuals...")
                clusters, clusters_res, eff, cost = global_method.explain_group(affected.drop(columns=['target']))

                sorted_actions_dict = dict(sorted(clusters_res.items(), key=lambda item: item[1]['cost']))
                actions = [stats["action"] for i,stats in sorted_actions_dict.items()]
                i=1
                all_clusters = {}
                num_features = test_data._get_numeric_data().columns.to_list()
                cate_features = test_data.columns.difference(num_features)

                for key in clusters:
                    clusters[key]['Cluster'] = i
                    all_clusters[i] = clusters[key]
                    i=i+1

                shared_resources["clusters_res"] = clusters_res
                combined_df = pd.concat(all_clusters.values(), ignore_index=True)
                cluster = combined_df['Cluster']
                combined_df = combined_df.drop(columns='Cluster')
                new_aff = affected.drop(columns='target')[:20].copy(deep=True)
                new_aff['unique_id'] = new_aff.groupby(list(new_aff.columns.difference(['index']))).cumcount()
                combined_df['unique_id'] = combined_df.groupby(list(combined_df.columns)).cumcount()
                result = combined_df.merge(new_aff, on=list(combined_df.columns) + ['unique_id'], how='left')
                result = result.drop(columns='unique_id')
                eff, cost, pred_list, chosen_actions, costs = cumulative(
                        model,
                        result,
                        actions,
                        global_method.dist_func_dataframe,
                        global_method.numerical_features_names,
                        global_method.categorical_features_names,
                        "-",
                    )
                
                eff_cost_actions = {}
                for i, arr in pred_list.items():
                    column_name = f"Action{i}_Prediction"
                    result[column_name] = arr
                    eff_act = pred_list[i].sum()/len(affected)
                    cost_act = costs[i-1][costs[i-1] != np.inf].sum()/pred_list[i].sum()
                    eff_cost_actions[i] = {'eff':eff_act , 'cost':cost_act}

                result['Cluster'] = cluster
                
                result['Chosen_Action'] = chosen_actions
                result['Chosen_Action'] = result['Chosen_Action'] + 1
                result = result.replace(np.inf , '-')
                shared_resources["affected_clusters"] = result


                filtered_data = {
                    k: {
                        **{
                            'action': {ak: av for ak, av in v['action'].items() if av != 0 and av != '-'}
                        },
                        **{kk: vv for kk, vv in v.items() if kk != 'action'}
                    }
                    for k, v in sorted_actions_dict.items()
                }
                actions_returned  = [stats["action"] for i,stats in filtered_data.items()]
                actions_ret = pd.DataFrame(actions_returned).fillna('-')
        
                return xai_service_pb2.ExplanationsResponse(
                    explainability_type = explanation_type,
                    explanation_method = 'global_counterfactuals',
                    explainability_model = model_path[0],
                    plot_name = 'Global Counterfactual Explanations',
                    plot_descr = "Counterfactual Explanations identify the minimal changes needed to alter a machine learning model's prediction for a given instance.",
                    plot_type = 'Table',
                    feature_list = dataset.drop(columns=['label']).columns.tolist(),
                    hyperparameter_list = [],
                    affected_clusters = {col: xai_service_pb2.TableContents(index=i+1,values=result[col].astype(str).tolist()) for i,col in enumerate(result.columns)},
                    eff_cost_actions = {
                        str(key): xai_service_pb2.EffCost(
                            eff=value['eff'],  
                            cost=value['cost']  
                        ) for key, value in eff_cost_actions.items()
                    },
                    TotalEffectiveness = float(round(eff/len(affected),2)),
                    TotalCost = float(round(cost/eff,2)),
                    actions = {col: xai_service_pb2.TableContents(index=i+1,values=actions_ret[col].astype(str).tolist()) for i,col in enumerate(actions_ret.columns)},

                ) 
            except UserConfigValidationException as e:
                # Handle known Dice error for missing counterfactuals
                if str(e) == "No counterfactuals found for any of the query points! Kindly check your configuration.":
                    return xai_service_pb2.ExplanationsResponse(
                    explainability_type=explanation_type,
                    explanation_method='global_counterfactuals',
                    explainability_model=model_path[0],
                    plot_name='Error',
                    plot_descr=f"An error occurred while generating the explanation: No counterfactuals found with the selected Local Counterfactual Method.",
                    plot_type='Error',
                    feature_list=[],
                    hyperparameter_list=[],
                    affected_clusters={},
                    eff_cost_actions={},
                    TotalEffectiveness=0.0,
                    TotalCost=0.0,
                    actions={},
                )

class PDPHandler(BaseExplanationHandler):

    def handle(self, request, explanation_type):

        if explanation_type == 'featureExplanation':
            # model_id = request.model_id
            model_path = request.model
            target = request.target_column

            train_data = _load_dataset(request.data.X_train)
            train_labels = _load_dataset(request.data.Y_train)  
           
            model, name = _load_model(model_path[0])


            if not request.feature1:
                logger.warning('Feature is missing, initializing with first feature from features list')
                features = train_data.columns.tolist()[0]
            else:
                features = request.feature1
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            numeric_features = train_data.select_dtypes(include=numerics).columns.tolist()
            categorical_features = train_data.columns.drop(numeric_features)

            pdp = partial_dependence(model, train_data, features = [train_data.columns.tolist().index(features)],
                                    feature_names=train_data.columns.tolist(),categorical_features=categorical_features)
            
            if type(pdp['grid_values'][0][0]) == str:
                axis_type='categorical' 
            else: axis_type = 'numerical'

            pdp_grid = [value.tolist() for value in pdp['grid_values']][0]
            pdp_vals = [value.tolist() for value in pdp['average']][0]
            return xai_service_pb2.ExplanationsResponse(
                explainability_type = explanation_type,
                explanation_method = 'pdp',
                explainability_model = model_path[0],
                plot_name = 'Partial Dependence Plot (PDP)',
                plot_descr = "PD (Partial Dependence) Plots show how a feature affects a model's predictions, holding other features constant, to illustrate feature impact.",
                plot_type = 'LinePlot',
                features = xai_service_pb2.Features(
                            feature1=features, 
                            feature2=''),
                feature_list = train_data.columns.tolist(),
                hyperparameter_list = [],
                xAxis = xai_service_pb2.Axis(
                            axis_name=f'{features}', 
                            axis_values=[str(value) for value in pdp_grid], 
                            axis_type=axis_type  
                ),
                yAxis = xai_service_pb2.Axis(
                            axis_name='PDP Values', 
                            axis_values=[str(value) for value in pdp_vals], 
                            axis_type='numerical'
                ),
                zAxis = xai_service_pb2.Axis(
                            axis_name='', 
                            axis_values='', 
                            axis_type=''                    
                )
            )
        elif explanation_type in ('hyperparameterExplanation', 'llmHyperparameterExplanation'):
            hyper_configs = request.hyper_configs
            hyper_space = create_hyperspace(hyper_configs)
            hyper_df, sorted_metrics = create_hyper_df(hyper_configs)

            logger.info('Training Surrogate Model...')
            surrogate_model = self._load_or_train_surrogate_model(hyper_df,sorted_metrics)
            logger.info("Trained Surrogate Model.")

            param_grid = transform_grid(hyper_space)
            param_space, name = dimensions_aslists(param_grid)
            space = Space(param_space)
            feats = {}
            for index,n in enumerate(name):
                feats[n] = index

            plot_dims = []
            for row in range(space.n_dims):
                # if space.dimensions[row].is_constant:
                #     continue
                plot_dims.append((row, space.dimensions[row]))

            pdp_samples = space.rvs(n_samples=1000,random_state=123456)
            if not request.feature1:
                logger.warning('Feature is missing, initializing with first hyperparameter from hyperparameters list')
                feature = name[0]
            else:
                feature = request.feature1

            xi = []
            yi=[]
            index, dim = plot_dims[feats[feature]]
            xi1, yi1 = partial_dependence_1D(space, surrogate_model,
                                                index,
                                                samples=pdp_samples,
                                                name=name,
                                                n_points=100)

            xi.append(xi1)
            yi.append(yi1)
                
            x = [arr.tolist() for arr in xi]
            y = [arr for arr in yi]
            axis_type = 'categorical' if isinstance(x[0][0], str) else 'numerical'
            return xai_service_pb2.ExplanationsResponse(
                explainability_type=explanation_type,
                explanation_method='pdp',
                explainability_model="",
                plot_name='Partial Dependence Plot (PDP)',
                plot_descr="PD (Partial Dependence) Plots show how different hyperparameter values affect a model's specified metric, holding other hyperparameters constant.",
                plot_type='LinePlot',
                features=xai_service_pb2.Features(
                    feature1=feature, 
                    feature2=''
                ),
                feature_list = [],
                hyperparameter_list = name,
                xAxis=xai_service_pb2.Axis(
                    axis_name=f'{feature}',
                    axis_values=[str(value) for value in x[0]],
                    axis_type=axis_type
                ),
                yAxis=xai_service_pb2.Axis(
                    axis_name='PDP Values',
                    axis_values=[str(value) for value in y[0]],
                    axis_type='numerical'
                ),
                zAxis=xai_service_pb2.Axis(
                    axis_name='',
                    axis_values='',
                    axis_type=''
                ),
            )
        elif explanation_type == 'experimentExplanation':
            experiment_configs = request.experiment_configs

            logger.debug("List of experiment configs received for PDP:")
            logger.debug(f"{experiment_configs=}")
            keep_common_variability_points(experiment_configs)

            hyper_space = create_hyperspace(experiment_configs)
            hyper_df, sorted_metrics = create_hyper_df(experiment_configs)

            logger.info('Training Surrogate Model...')
            surrogate_model = self._load_or_train_surrogate_model(hyper_df, sorted_metrics)
            logger.info("Trained Surrogate Model.")
            
            param_grid = transform_grid(hyper_space)
            param_space, name = dimensions_aslists(param_grid)
            space = Space(param_space)
            feats = {}
            for index,n in enumerate(name):
                feats[n] = index

            plot_dims = []
            for row in range(space.n_dims):
                # if space.dimensions[row].is_constant:
                #     continue
                plot_dims.append((row, space.dimensions[row]))
                
            pdp_samples = space.rvs(n_samples=1000,random_state=123456)
            if not request.feature1:
                logger.warning('Feature is missing, initializing with first hyperparameter from hyperparameters list')
                feature = name[0]
            else: 
                feature = request.feature1

            xi = []
            yi=[]
            index, dim = plot_dims[feats[feature]]
            xi1, yi1 = partial_dependence_1D(space, surrogate_model,
                                                index,
                                                samples=pdp_samples,
                                                name=name,
                                                n_points=100)

            xi.append(xi1)
            yi.append(yi1)
                
            x = [arr.tolist() for arr in xi]
            y = [arr for arr in yi]
            axis_type = 'categorical' if isinstance(x[0][0], str) else 'numerical'
            return xai_service_pb2.ExplanationsResponse(
                explainability_type=explanation_type,
                explanation_method='pdp',
                explainability_model="",
                plot_name='Partial Dependence Plot (PDP)',
                plot_descr="PD (Partial Dependence) Plots show how different hyperparameter values affect a model's specified metric, holding other hyperparameters constant.",
                plot_type='LinePlot',
                features=xai_service_pb2.Features(
                    feature1=feature, 
                    feature2=''
                ),
                feature_list = [],
                hyperparameter_list = name,
                xAxis=xai_service_pb2.Axis(
                    axis_name=f'{feature}',
                    axis_values=[str(value) for value in x[0]],
                    axis_type=axis_type
                ),
                yAxis=xai_service_pb2.Axis(
                    axis_name='PDP Values',
                    axis_values=[str(value) for value in y[0]],
                    axis_type='numerical'
                ),
                zAxis=xai_service_pb2.Axis(
                    axis_name='',
                    axis_values='',
                    axis_type=''
                ),
            )
        else:
            raise ValueError(f"Unknown explanation type: {explanation_type}")

class TwoDPDPHandler(BaseExplanationHandler):

    def handle(self, request, explanation_type):
        if explanation_type == 'featureExplanation':
            model_path = request.model
            target = request.target_column
            train_data = _load_dataset(request.data.X_train)
            train_labels = _load_dataset(request.data.Y_train)  

            model, name = _load_model(model_path[0])
     
                             
            if not request.feature1:
                logger.warning('Feature is missing, initializing with first feature from features list')
                feature1 = train_data.columns.tolist()[0]
                feature2 = train_data.columns.tolist()[1]
            else: 
                feature1 = request.feature1
                feature2 = request.feature2
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

            numeric_features = train_data.select_dtypes(include=numerics).columns.tolist()
            categorical_features = train_data.columns.drop(numeric_features)

            pdp = partial_dependence(model, train_data, features = [(train_data.columns.tolist().index(feature1),train_data.columns.tolist().index(feature2))],
                                    feature_names=train_data.columns.tolist(),categorical_features=categorical_features,grid_resolution=25)
            

            if type(pdp['grid_values'][0][0]) == str:
                axis_type_0='categorical' 
            else: axis_type_0 = 'numerical'

            if type(pdp['grid_values'][1][0]) == str:
                axis_type_1='categorical' 
            else: axis_type_1 = 'numerical'


            pdp_grid_1 = [value.tolist() for value in pdp['grid_values']][0]
            pdp_grid_2 = [value.tolist() for value in pdp['grid_values']][1]
            pdp_vals = [value.tolist() for value in pdp['average']][0]
            return xai_service_pb2.ExplanationsResponse(
                explainability_type = explanation_type,
                explanation_method = '2dpdp',
                explainability_model = model_path[0],
                plot_name = '2D-Partial Dependence Plot (2D-PDP)',
                plot_descr = "2D-PD (Partial Dependence) plots showcase the marginal effect of two features on a model's predictions while averaging out the effects of all other features.",
                plot_type = 'ContourPlot',
                features = xai_service_pb2.Features(
                            feature1=feature1, 
                            feature2=feature2),
                feature_list = train_data.columns.tolist(),
                hyperparameter_list = [],
                xAxis = xai_service_pb2.Axis(
                            axis_name=f'{feature1}', 
                            axis_values=[str(value) for value in pdp_grid_1], 
                            axis_type=axis_type_0  
                ),
                yAxis = xai_service_pb2.Axis(
                            axis_name=f'{feature2}', 
                            axis_values=[str(value) for value in pdp_grid_2], 
                            axis_type=axis_type_1
                ),
                zAxis = xai_service_pb2.Axis(
                            axis_name='', 
                            axis_values=[str(value) for value in pdp_vals], 
                            axis_type='numerical'                    
                ),
            )
        elif explanation_type in ('hyperparameterExplanation', 'llmHyperparameterExplanation'):
            hyper_configs = request.hyper_configs
            hyper_space = create_hyperspace(hyper_configs)
            hyper_df,sorted_metrics = create_hyper_df(hyper_configs)

            logger.info('Training Surrogate Model...')
            surrogate_model = self._load_or_train_surrogate_model(hyper_df,sorted_metrics)
            logger.info("Trained Surrogate Model.")
            
            param_grid = transform_grid(hyper_space)
            param_space, name = dimensions_aslists(param_grid)
            space = Space(param_space)
            if not request.feature1:
                logger.warning('Feature is missing, initializing with first hyperparameter from hyperparameters list')
                feature1 = name[0]
                feature2 = name[1]
            else: 
                feature1 = request.feature1
                feature2 = request.feature2
            
            index1 = name.index(feature1)
            index2 = name.index(feature2)


            plot_dims = []
            for row in range(space.n_dims):
                if space.dimensions[row].is_constant:
                    continue
                plot_dims.append((row, space.dimensions[row]))
            
            pdp_samples = space.rvs(n_samples=1000,random_state=123456)

            _ ,dim_1 = plot_dims[index1]
            _ ,dim_2 = plot_dims[index2]
            xi, yi, zi = partial_dependence_2D(space, surrogate_model,
                                                    index1, index2,
                                                    pdp_samples,name, 100)
            
            
            x = [arr.tolist() for arr in xi]
            y = [arr.tolist() for arr in yi]
            z = [arr.tolist() for arr in zi]

            return xai_service_pb2.ExplanationsResponse(
                        explainability_type = explanation_type,
                        explanation_method = '2dpdp',
                        explainability_model = '',
                        plot_name = '2D-Partial Dependence Plot (2D-PDP)',
                        plot_descr = "2D-PD plots visualize how the model's specified metric changes when two hyperparameters vary.",
                        plot_type = 'ContourPlot',
                        features = xai_service_pb2.Features(
                                    feature1=feature1, 
                                    feature2=feature2),
                        feature_list = [],
                        hyperparameter_list = name,
                        xAxis = xai_service_pb2.Axis(
                                    axis_name=f'{feature2}', 
                                    axis_values=[str(value) for value in x], 
                                    axis_type='categorical' if isinstance(x[0], str) else 'numerical'
                        ),
                        yAxis = xai_service_pb2.Axis(
                                    axis_name=f'{feature1}', 
                                    axis_values=[str(value) for value in y], 
                                    axis_type='categorical' if isinstance(y[0], str) else 'numerical'
                        ),
                        zAxis = xai_service_pb2.Axis(
                                    axis_name='', 
                                    axis_values=[str(value) for value in z], 
                                    axis_type='numerical' 
                        ),
                        
            )
        
        elif explanation_type == 'experimentExplanation':
            experiment_configs = request.experiment_configs

            keep_common_variability_points(experiment_configs)

            hyper_space = create_hyperspace(experiment_configs)
            hyper_df,sorted_metrics = create_hyper_df(experiment_configs)

            logger.info('Training Surrogate Model...')
            surrogate_model = self._load_or_train_surrogate_model(hyper_df,sorted_metrics)
            logger.info("Trained Surrogate Model.")
            
            param_grid = transform_grid(hyper_space)
            param_space, name = dimensions_aslists(param_grid)
            space = Space(param_space)
            if not request.feature1:
                logger.warning('Feature is missing, initializing with first hyperparameter from hyperparameters list')
                feature1 = name[0]
                feature2 = name[1]
            else: 
                feature1 = request.feature1
                feature2 = request.feature2
            logger.info(f"Generating 2D-PDP for features: {feature1} and {feature2}")

            index1 = name.index(feature1)
            index2 = name.index(feature2)
            logger.info(f"Feature indices are: {index1} and {index2}")


            # plot_dims = []
            # for row in range(space.n_dims):
            #     if space.dimensions[row].is_constant:
            #         continue
            #     plot_dims.append((row, space.dimensions[row]))
            # #logger.info(f"Plot dimensions identified: {plot_dims}")
            
            pdp_samples = space.rvs(n_samples=1000,random_state=123456)

            # _ ,dim_1 = plot_dims[index1]
            # _ ,dim_2 = plot_dims[index2]
            xi, yi, zi = partial_dependence_2D(space, surrogate_model,
                                                    index1, index2,
                                                    pdp_samples,name, 100)
            
            
            x = [arr.tolist() for arr in xi]
            y = [arr.tolist() for arr in yi]
            z = [arr.tolist() for arr in zi]

            return xai_service_pb2.ExplanationsResponse(
                        explainability_type = explanation_type,
                        explanation_method = '2dpdp',
                        explainability_model = '',
                        plot_name = '2D-Partial Dependence Plot (2D-PDP)',
                        plot_descr = "2D-PD plots visualize how the model's specified metric changes when two hyperparameters vary.",
                        plot_type = 'ContourPlot',
                        features = xai_service_pb2.Features(
                                    feature1=feature1, 
                                    feature2=feature2),
                        feature_list = [],
                        hyperparameter_list = name,
                        xAxis = xai_service_pb2.Axis(
                                    axis_name=f'{feature2}', 
                                    axis_values=[str(value) for value in x],
                                    axis_type='categorical' if isinstance(x[0], str) else 'numerical'
                        ),
                        yAxis = xai_service_pb2.Axis(
                                    axis_name=f'{feature1}',
                                    axis_values=[str(value) for value in y],
                                    axis_type='categorical' if isinstance(y[0], str) else 'numerical'
                        ),
                        zAxis = xai_service_pb2.Axis(
                                    axis_name='',
                                    axis_values=[str(value) for value in z],
                                    axis_type='numerical'
                        ),
            )
        else:
            raise ValueError(f"Unknown explanation type: {explanation_type}")
    
class ALEHandler(BaseExplanationHandler):

    def handle(self, request, explanation_type):
        if explanation_type == 'featureExplanation':
            model_path = request.model
            target = request.target_column
            train_data = _load_dataset(request.data.X_train)
            train_labels = _load_dataset(request.data.Y_train)  


            model, name = _load_model(model_path[0])

            if not request.feature1:
                logger.warning('Feature is missing, initializing with first features from features list')
                features = train_data.columns.tolist()[0]
            else: 
                features = request.feature1

            if train_data[features].dtype in ['int','float']:
                ale_eff = ale(X=train_data, model=model, feature=[features],plot=False, grid_size=50, include_CI=True, C=0.95)
            else:
                ale_eff = ale(X=train_data, model=model, feature=[features],plot=False, grid_size=50, predictors=train_data.columns.tolist(), include_CI=True, C=0.95)

            return xai_service_pb2.ExplanationsResponse(
                explainability_type = explanation_type,
                explanation_method = 'ale',
                explainability_model = model_path[0],
                plot_name = 'Accumulated Local Effects Plot (ALE)',
                plot_descr = "ALE plots illustrate the effect of a single feature on the predicted outcome of a machine learning model.",
                plot_type = 'LinePLot',
                features = xai_service_pb2.Features(
                            feature1=features, 
                            feature2=''),
                feature_list = train_data.columns.tolist(),
                hyperparameter_list = [],
                xAxis = xai_service_pb2.Axis(
                            axis_name=f'{features}', 
                            axis_values=[str(value) for value in ale_eff.index.tolist()], 
                            axis_type='categorical' if isinstance(ale_eff.index.tolist()[0], str) else 'numerical'
                ),
                yAxis = xai_service_pb2.Axis(
                            axis_name='ALE Values', 
                            axis_values=[str(value) for value in ale_eff.eff.tolist()], 
                            axis_type='categorical' if isinstance(ale_eff.eff.tolist()[0], str) else 'numerical'
                ),
                zAxis = xai_service_pb2.Axis(
                            axis_name='', 
                            axis_values='', 
                            axis_type=''                    
                ),
                
            )
        elif explanation_type in ('hyperparameterExplanation', 'llmHyperparameterExplanation'):
            hyper_configs = request.hyper_configs
            hyper_space = create_hyperspace(hyper_configs)
            hyper_df,sorted_metrics = create_hyper_df(hyper_configs)

            logger.info('Training Surrogate Model')

            surrogate_model = self._load_or_train_surrogate_model(hyper_df,sorted_metrics)

            param_grid = transform_grid(hyper_space)
            param_space, name = dimensions_aslists(param_grid)
            space = Space(param_space)

            plot_dims = []
            for row in range(space.n_dims):
                if space.dimensions[row].is_constant:
                    continue
                plot_dims.append((row, space.dimensions[row]))

            if not request.feature1:
                logger.warning('Feature is missing, initializing with first hyperparameter from hyperparameter list')
                feature1 = name[0]
            else: 
                feature1 = request.feature1

            pdp_samples = space.rvs(n_samples=1000,random_state=123456)
            data = pd.DataFrame(pdp_samples,columns=[n for n in name])

            if data[feature1].dtype in ['int','float']: 
                ale_eff = ale(X=data, model=surrogate_model, feature=[feature1],plot=False, grid_size=50, include_CI=True, C=0.95)
            else:
                ale_eff = ale(X=data, model=surrogate_model, feature=[feature1],plot=False, grid_size=50,predictors=data.columns.tolist(), include_CI=True, C=0.95)
            return xai_service_pb2.ExplanationsResponse(
                explainability_type = explanation_type,
                explanation_method = 'ale',
                explainability_model = '',
                plot_name = 'Accumulated Local Effects Plot (ALE)',
                plot_descr = "ALE Plots illustrate the effect of a single hyperparameter on the accuracy of a machine learning model.",
                plot_type = 'LinePLot',
                features = xai_service_pb2.Features(
                            feature1=feature1, 
                            feature2=''),
                feature_list = [],            
                hyperparameter_list = name,
                xAxis = xai_service_pb2.Axis(
                            axis_name=f'{feature1}', 
                            axis_values=[str(value) for value in ale_eff.index.tolist()], 
                            axis_type='categorical' if isinstance(ale_eff.index.tolist()[0], str) else 'numerical'
                ),
                yAxis = xai_service_pb2.Axis(
                            axis_name='ALE Values', 
                            axis_values=[str(value) for value in ale_eff.eff.tolist()], 
                            axis_type='categorical' if isinstance(ale_eff.eff.tolist()[0], str) else 'numerical'
                ),
                zAxis = xai_service_pb2.Axis(
                            axis_name='', 
                            axis_values='', 
                            axis_type=''                    
                ),
                
            )
        
        elif explanation_type == 'experimentExplanation':
            experiment_configs = request.experiment_configs

            keep_common_variability_points(experiment_configs)

            hyper_space = create_hyperspace(experiment_configs)
            hyper_df,sorted_metrics = create_hyper_df(experiment_configs)

            logger.info('Training Surrogate Model...')
            surrogate_model = self._load_or_train_surrogate_model(hyper_df,sorted_metrics)
            logger.info("Trained Surrogate Model.")

            param_grid = transform_grid(hyper_space)
            param_space, name = dimensions_aslists(param_grid)
            space = Space(param_space)

            plot_dims = []
            for row in range(space.n_dims):
                if space.dimensions[row].is_constant:
                    continue
                plot_dims.append((row, space.dimensions[row]))

            if not request.feature1:
                logger.warning('Feature is missing, initializing with first hyperparameter from hyperparameter list')
                feature1 = name[0]
            else: 
                feature1 = request.feature1

            pdp_samples = space.rvs(n_samples=1000,random_state=123456)
            data = pd.DataFrame(pdp_samples,columns=[n for n in name])

            if data[feature1].dtype in ['int','float']: 
                ale_eff = ale(X=data, model=surrogate_model, feature=[feature1],plot=False, grid_size=50, include_CI=True, C=0.95)
            else:
                ale_eff = ale(X=data, model=surrogate_model, feature=[feature1],plot=False, grid_size=50,predictors=data.columns.tolist(), include_CI=True, C=0.95)
            return xai_service_pb2.ExplanationsResponse(
                explainability_type = explanation_type,
                explanation_method = 'ale',
                explainability_model = '',
                plot_name = 'Accumulated Local Effects Plot (ALE)',
                plot_descr = "ALE Plots illustrate the effect of a single hyperparameter on the accuracy of a machine learning model.",
                plot_type = 'LinePLot',
                features = xai_service_pb2.Features(
                            feature1=feature1, 
                            feature2=''),
                feature_list = [],            
                hyperparameter_list = name,
                xAxis = xai_service_pb2.Axis(
                            axis_name=f'{feature1}', 
                            axis_values=[str(value) for value in ale_eff.index.tolist()], 
                            axis_type='categorical' if isinstance(ale_eff.index.tolist()[0], str) else 'numerical'
                ),
                yAxis = xai_service_pb2.Axis(
                            axis_name='ALE Values',
                            axis_values=[str(value) for value in ale_eff.eff.tolist()],
                            axis_type='categorical' if isinstance(ale_eff.eff.tolist()[0], str) else 'numerical'
                ),
                zAxis = xai_service_pb2.Axis(
                            axis_name='',
                            axis_values='',
                            axis_type=''
                ),
            )
        
        else:
            raise ValueError(f"Unknown explanation type: {explanation_type}")
        
class CounterfactualsHandler(BaseExplanationHandler):

    def handle(self, request, explanation_type):
        
        if explanation_type == 'featureExplanation':
            query = request.query
            query = ast.literal_eval(query)
            query = pd.DataFrame([query])

            query = query.drop(columns=['label'])
            target = 'label'

            model_path = request.model
            data_path = request.data
            train_data = _load_dataset(request.data.X_train)
            train_labels = _load_dataset(request.data.Y_train)  
            model, name = _load_model(model_path[0])

            
            dataframe = pd.concat([train_data.reset_index(drop=True), train_labels.reset_index(drop=True)], axis = 1)

            d = dice_ml.Data(dataframe=dataframe, 
                continuous_features=dataframe.drop(columns=target).select_dtypes(include='number').columns.tolist()
                , outcome_name=target)
    
            # Using sklearn backend
            m = dice_ml.Model(model=model, backend="sklearn")
            # Using method=random for generating CFs
            exp = dice_ml.Dice(d, m, method="random")
            try:
                e1 = exp.generate_counterfactuals(query.drop(columns=['prediction']), total_CFs=5, desired_class="opposite",sample_size=5000)
                e1.visualize_as_dataframe(show_only_changes=True)
                cfs = e1.cf_examples_list[0].final_cfs_df
                query.rename(columns={"prediction": target},inplace=True)

                cfs['Type'] = 'Counterfactual'
                query['Type'] = 'Factual'
                factual = query.iloc[0].drop(['Type']) if 'Type' in query.columns else query.iloc[0]
                

                # Compute differences
                diffs = []
                for i, row in cfs.iterrows():
                    diff_row = {}
                    for col in factual.index:
                        if col in ['Type', target]:  # skip label and type
                            continue
                        cf_val = row[col]
                        f_val = factual[col]
                        if pd.isna(cf_val) or pd.isna(f_val):
                            diff = '-'
                        elif cf_val != f_val:
                            try:
                                # For numerical changes
                                delta = cf_val - f_val
                                diff = f'+{delta}' if delta > 0 else f'{delta}'
                            except:
                                # For categorical or non-subtractable values
                                diff = cf_val
                        else:
                            diff = '-'
                        diff_row[col] = diff
                    diff_row['Type'] = 'Counterfactual'
                    diffs.append(diff_row)

                # Convert differences to DataFrame
                diffs_df = pd.DataFrame(diffs)

                # Append factual row (unaltered)
                factual_diff = {col: factual[col] for col in factual.index if col not in ['Type', target]}
                factual_diff['Type'] = 'Factual'
                diffs_df = pd.concat([pd.DataFrame([factual_diff]), diffs_df], ignore_index=True)
                # Drop columns where all counterfactual rows are '-'
                cf_only = diffs_df[diffs_df['Type'] == 'Counterfactual']
                cols_to_drop = [col for col in cf_only.columns if col not in ['Type'] and (cf_only[col] == '-').all()]
                diffs_df.drop(columns=cols_to_drop, inplace=True)
                cfs = pd.concat([query,cfs])
                diffs_df['label'] = cfs['label'].values

                logger.debug("Differences DataFrame:")
                logger.debug(diffs_df)


                return xai_service_pb2.ExplanationsResponse(
                    explainability_type = explanation_type,
                    explanation_method = 'counterfactuals',
                    explainability_model = model_path[0],
                    plot_name = 'Counterfactual Explanations',
                    plot_descr = "Counterfactual Explanations identify the minimal changes needed to alter a machine learning model's prediction for a given instance.",
                    plot_type = 'Table',
                    feature_list = dataframe.columns.tolist(),
                    hyperparameter_list = [],
                    table_contents = {col: xai_service_pb2.TableContents(index=i+1,values=diffs_df[col].astype(str).tolist()) for i,col in enumerate(diffs_df.columns)}
                )
            except UserConfigValidationException as e:
                # Handle known Dice error for missing counterfactuals
                if str(e) == "No counterfactuals found for any of the query points! Kindly check your configuration.":
                    return xai_service_pb2.ExplanationsResponse(
                    explainability_type=explanation_type,
                    explanation_method='couterfactuals',
                    explainability_model=model_path[0],
                    plot_name='Error',
                    plot_descr=f"An error occurred while generating the explanation: {str(e)}",
                    plot_type='Error',
                    feature_list=dataframe.columns.tolist(),
                    hyperparameter_list=[],
                )
        elif explanation_type == 'hyperparameterExplanation':
            model_path = request.model
            logger.debug(f"{model_path=}")
            hyper_configs = request.hyper_configs
            query = request.query
            
            query = ast.literal_eval(query)
            if type(query) == dict:
                query = pd.DataFrame([query])
                prediction = query['prediction']
                label = query['label']
                query = query.drop(columns=['label','prediction'])
            else:
                query = np.array(query)
                label = pd.Series(1)
                prediction = pd.Series(2)

            logger.info('Creating Proxy Dataset and Model')
            try:
                surrogate_model , proxy_dataset = self._load_or_train_cf_surrogate_model(hyper_configs,query)
            except (UserConfigValidationException, ValueError) as e:
            # Handle known Dice error for missing counterfactuals
                return xai_service_pb2.ExplanationsResponse(
                explainability_type=explanation_type,
                explanation_method='couterfactuals',
                explainability_model=model_path[0],
                plot_name='Error',
                plot_descr=f"An error occurred while generating the explanation: {str(e)}",
                plot_type='Error',
                feature_list=[],
                hyperparameter_list=[],
            )
            hp_query = create_cfquery_df(hyper_configs,model_path[0])

            d = dice_ml.Data(dataframe=proxy_dataset, continuous_features=proxy_dataset.drop(columns='BinaryLabel').select_dtypes(include='number').columns.tolist(), outcome_name='BinaryLabel')
            m = dice_ml.Model(model=surrogate_model, backend="sklearn")
            exp = dice_ml.Dice(d, m, method="random")

            try:
                e1 = exp.generate_counterfactuals(hp_query, total_CFs=5, desired_class=int(label.values[0]),sample_size=5000)
            except UserConfigValidationException as e:
            # Handle known Dice error for missing counterfactuals
                return xai_service_pb2.ExplanationsResponse(
                explainability_type=explanation_type,
                explanation_method='couterfactuals',
                explainability_model=model_path[0],
                plot_name='Error',
                plot_descr=f"An error occurred while generating the explanation: {str(e)}",
                plot_type='Error',
                feature_list=[],
                hyperparameter_list=hp_query.columns.tolist(),
            )

            dtypes_dict = proxy_dataset.drop(columns='BinaryLabel').dtypes.to_dict()
            cfs = e1.cf_examples_list[0].final_cfs_df
            logger.debug("Counterfactuals DataFrame:")
            logger.debug(cfs)
            for col, dtype in dtypes_dict.items():
                cfs[col] = cfs[col].astype(dtype)
                scaled_query, scaled_cfs = min_max_scale(proxy_dataset=proxy_dataset,factual=hp_query.copy(deep=True),counterfactuals=cfs.copy(deep=True),label='BinaryLabel')
            cfs['Cost'] = cf_difference(scaled_query, scaled_cfs)
            cfs = cfs.sort_values(by='Cost')
            cfs['Type'] = 'Counterfactual'
            hp_query['Cost'] = '-'
            hp_query['Type'] = 'Factual'

            hp_query['BinaryLabel'] = prediction
            logger.debug(f"{type(prediction)=}")
            logger.debug(f"{prediction.values=}")
            #cfs['BinaryLabel'] = 1 if prediction.values == 0 else 0
            # Compute differences only for changed features
            factual = hp_query.iloc[0].drop(['Type', 'Cost', 'BinaryLabel'])
            diffs = []

            for _, row in cfs.iterrows():
                diff_row = {}
                for col in factual.index:
                    cf_val = row[col]
                    f_val = factual[col]
                    if pd.isna(cf_val) or pd.isna(f_val):
                        diff = '-'
                    elif cf_val != f_val:
                        try:
                            delta = cf_val - f_val
                            diff = f'+{delta}' if delta > 0 else f'{delta}'
                        except:
                            diff = cf_val
                    else:
                        diff = '-'
                    diff_row[col] = diff
                diff_row['Cost'] = row['Cost']
                diff_row['Type'] = 'Counterfactual'
                diff_row['BinaryLabel'] = row['BinaryLabel']
                diffs.append(diff_row)

            # Build DataFrame
            diffs_df = pd.DataFrame(diffs)
            

            # Add factual row
            factual_diff = {col: factual[col] for col in factual.index}
            factual_diff['Cost'] = '-'
            factual_diff['Type'] = 'Factual'
            factual_diff['BinaryLabel'] = prediction.values[0]
            diffs_df = pd.concat([pd.DataFrame([factual_diff]), diffs_df], ignore_index=True)

            # Drop unchanged columns
            cf_only = diffs_df[diffs_df['Type'] == 'Counterfactual']
            cols_to_drop = [col for col in factual.index if (cf_only[col] == '-').all()]
            diffs_df.drop(columns=cols_to_drop, inplace=True)
            logger.debug("Differences DataFrame:")
            logger.debug(diffs_df)



            return xai_service_pb2.ExplanationsResponse(
                explainability_type = explanation_type,
                explanation_method = 'counterfactuals',
                explainability_model = model_path[0],
                plot_name = 'Counterfactual Explanations',
                plot_descr = "Counterfactual Explanations identify the minimal changes on hyperparameter values in order to correctly classify a given missclassified instance.",
                plot_type = 'Table',
                feature_list = [],
                hyperparameter_list = hp_query.drop(columns=['Cost','Type','BinaryLabel']).columns.tolist(),
                table_contents = {col: xai_service_pb2.TableContents(index=i+1,values=diffs_df[col].astype(str).tolist()) for i,col in enumerate(diffs_df.columns)}
            )
        
        elif explanation_type == 'experimentExplanation':
            raise NotImplementedError("Counterfactual explanations for experimentExplanation type are not fully implemented yet.")
            
            experiment_configs = request.experiment_configs
            
            # keep_common_variability_points(experiment_configs)

            query = request.query
            
            query = ast.literal_eval(query)
            if type(query) == dict:
                query = pd.DataFrame([query])
                prediction = query['prediction']
                label = query['label']
                query = query.drop(columns=['label','prediction'])
            else:
                query = np.array(query)
                label = pd.Series(1)
                prediction = pd.Series(2)

            logger.info('Creating Proxy Dataset and Model')
            try:
                surrogate_model , proxy_dataset = self._load_or_train_cf_surrogate_model(experiment_configs,query)
            except (UserConfigValidationException, ValueError) as e:
            # Handle known Dice error for missing counterfactuals
                return xai_service_pb2.ExplanationsResponse(
                explainability_type=explanation_type,
                explanation_method='couterfactuals',
                explainability_model='',
                plot_name='Error',
                plot_descr=f"An error occurred while generating the explanation: {str(e)}",
                plot_type='Error',
                feature_list=[],
                hyperparameter_list=[],
            )
            
            hp_query = create_cfquery_df(hyper_configs,model_path[0])

            d = dice_ml.Data(dataframe=proxy_dataset, continuous_features=proxy_dataset.drop(columns='BinaryLabel').select_dtypes(include='number').columns.tolist(), outcome_name='BinaryLabel')
            m = dice_ml.Model(model=surrogate_model, backend="sklearn")
            exp = dice_ml.Dice(d, m, method="random")

            try:
                e1 = exp.generate_counterfactuals(hp_query, total_CFs=5, desired_class=int(label.values[0]),sample_size=5000)
            except UserConfigValidationException as e:
            # Handle known Dice error for missing counterfactuals
                return xai_service_pb2.ExplanationsResponse(
                explainability_type=explanation_type,
                explanation_method='couterfactuals',
                explainability_model=model_path[0],
                plot_name='Error',
                plot_descr=f"An error occurred while generating the explanation: {str(e)}",
                plot_type='Error',
                feature_list=[],
                hyperparameter_list=hp_query.columns.tolist(),
            )

            dtypes_dict = proxy_dataset.drop(columns='BinaryLabel').dtypes.to_dict()
            cfs = e1.cf_examples_list[0].final_cfs_df
            logger.debug("Counterfactuals DataFrame:")
            logger.debug(cfs)
            for col, dtype in dtypes_dict.items():
                cfs[col] = cfs[col].astype(dtype)
                scaled_query, scaled_cfs = min_max_scale(proxy_dataset=proxy_dataset,factual=hp_query.copy(deep=True),counterfactuals=cfs.copy(deep=True),label='BinaryLabel')
            cfs['Cost'] = cf_difference(scaled_query, scaled_cfs)
            cfs = cfs.sort_values(by='Cost')
            cfs['Type'] = 'Counterfactual'
            hp_query['Cost'] = '-'
            hp_query['Type'] = 'Factual'

            hp_query['BinaryLabel'] = prediction
            logger.debug(f"{type(prediction)=}")
            logger.debug(f"{prediction.values=}")
            #cfs['BinaryLabel'] = 1 if prediction.values == 0 else 0
            # Compute differences only for changed features
            factual = hp_query.iloc[0].drop(['Type', 'Cost', 'BinaryLabel'])
            diffs = []

            for _, row in cfs.iterrows():
                diff_row = {}
                for col in factual.index:
                    cf_val = row[col]
                    f_val = factual[col]
                    if pd.isna(cf_val) or pd.isna(f_val):
                        diff = '-'
                    elif cf_val != f_val:
                        try:
                            delta = cf_val - f_val
                            diff = f'+{delta}' if delta > 0 else f'{delta}'
                        except:
                            diff = cf_val
                    else:
                        diff = '-'
                    diff_row[col] = diff
                diff_row['Cost'] = row['Cost']
                diff_row['Type'] = 'Counterfactual'
                diff_row['BinaryLabel'] = row['BinaryLabel']
                diffs.append(diff_row)

            # Build DataFrame
            diffs_df = pd.DataFrame(diffs)
            

            # Add factual row
            factual_diff = {col: factual[col] for col in factual.index}
            factual_diff['Cost'] = '-'
            factual_diff['Type'] = 'Factual'
            factual_diff['BinaryLabel'] = prediction.values[0]
            diffs_df = pd.concat([pd.DataFrame([factual_diff]), diffs_df], ignore_index=True)

            # Drop unchanged columns
            cf_only = diffs_df[diffs_df['Type'] == 'Counterfactual']
            cols_to_drop = [col for col in factual.index if (cf_only[col] == '-').all()]
            diffs_df.drop(columns=cols_to_drop, inplace=True)
            logger.debug("Differences DataFrame:")
            logger.debug(diffs_df)



            return xai_service_pb2.ExplanationsResponse(
                explainability_type = explanation_type,
                explanation_method = 'counterfactuals',
                explainability_model = model_path[0],
                plot_name = 'Counterfactual Explanations',
                plot_descr = "Counterfactual Explanations identify the minimal changes on hyperparameter values in order to correctly classify a given missclassified instance.",
                plot_type = 'Table',
                feature_list = [],
                hyperparameter_list = hp_query.drop(columns=['Cost','Type','BinaryLabel']).columns.tolist(),
                table_contents = {col: xai_service_pb2.TableContents(index=i+1,values=diffs_df[col].astype(str).tolist()) for i,col in enumerate(diffs_df.columns)}
            )

class PrototypesHandler(BaseExplanationHandler):

    def handle(self, request, explanation_type):
        query = request.query
        query = ast.literal_eval(query)
        query = pd.DataFrame([query])
        label = query['label']
        prediction = query['prediction']
        query.drop(columns=['label','prediction'],inplace=True)
        categorical_cols = query.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = query.select_dtypes(exclude=['object', 'category']).columns.tolist()

        model_path = request.model
        train_data = _load_dataset(request.data.X_train)
        train_labels = _load_dataset(request.data.Y_train)  

        model, name = _load_model(model_path[0])

        # label = train_data[target]
        # train_data = train_data.drop(columns=[target])

        # mask = ~train_data.eq(query.iloc[0]).all(axis=1)
        # train_data = train_data[mask]


        # train_data[target] = model.predict(train_data)
        
        # query['prediction'] = model.predict(query)
        # print(query)
        explainer = ProtodashExplainer()
        reference_set_train = train_data.copy(deep=True)
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(reference_set_train[categorical_cols])
        encoded_cat_feature_names = encoder.get_feature_names_out(categorical_cols)


        query_encoded_cat = encoder.transform(query[categorical_cols])
        ref_encoded_cat = encoder.transform(reference_set_train[categorical_cols])
        query_numeric = query[numerical_cols].to_numpy()
        ref_numeric = reference_set_train[numerical_cols].to_numpy()

        query_encoded = np.hstack((query_numeric, query_encoded_cat))
        ref_encoded = np.hstack((ref_numeric, ref_encoded_cat))

        #[test_data[target]==query['prediction'].values[0]].drop(columns=[target])

        (W, S, _)= explainer.explain(query_encoded.reshape(1, -1), ref_encoded, m=5)
        prototypes = reference_set_train.reset_index(drop=True).iloc[S, :].copy()
        logger.debug(f"{type(prototypes)=}")
        # prototypes.rename(columns={target:'label'},inplace=True)
        prototypes['prediction'] =  model.predict(prototypes)
        prototypes = prototypes.reset_index(drop=True).T
        prototypes.rename(columns={0:'Prototype1',1:'Prototype2',2:'Prototype3',3:'Prototype4',4:'Prototype5'},inplace=True)
        prototypes = prototypes.reset_index()

        prototypes.set_index('index', inplace=True)

        # Create a new empty dataframe for boolean results
        boolean_df = pd.DataFrame(index=prototypes.index)
        query['prediction'] = prediction

        # Iterate over each column and compare with the series
        for col in prototypes.columns:
            boolean_df[col] = prototypes[col] == query.loc[0][prototypes.index].values

        prototypes.reset_index(inplace=True)
        new_row = pd.DataFrame([{
            'index': 'Weights',
            'Prototype1': np.around(W/np.sum(W), 2)[0],
            'Prototype2': np.around(W/np.sum(W), 2)[1],
            'Prototype3': np.around(W/np.sum(W), 2)[2],
            'Prototype4': np.around(W/np.sum(W), 2)[3],
            'Prototype5': np.around(W/np.sum(W), 2)[4]
        }])

        # Concatenate it to the original DataFrame
        prototypes = pd.concat([prototypes, new_row], ignore_index=True)
        new_bool_row = pd.DataFrame([{
            'index': 'Weights',
            'Prototype1': False,
            'Prototype2': False,
            'Prototype3': False,
            'Prototype4': False,
            'Prototype5': False
        }])
        boolean_df = pd.concat([boolean_df, new_bool_row], ignore_index=True)

        logger.debug("Prototypes DataFrame:")
        logger.debug(prototypes)
        # Create table_contents dictionary for prototypes
        table_contents =  {col: xai_service_pb2.TableContents(index=i+1,values=prototypes[col].astype(str).tolist(),colour =boolean_df[col].astype(str).tolist()) for i,col in enumerate(prototypes.columns)}


        return xai_service_pb2.ExplanationsResponse(
            explainability_type = explanation_type,
            explanation_method = 'prototypes',
            explainability_model = model_path[0],
            plot_name = 'Prototypes',
            plot_descr = "Prototypes are prototypical examples that capture the underlying distribution of a dataset. It also weights each prototype to quantify how well it represents the data.",
            plot_type = 'Table',
            feature_list = train_data.columns.tolist(),
            hyperparameter_list = [],
            table_contents = table_contents
        )

class SegmentationAttributionHandler(BaseExplanationHandler):
    """Handler that computes attributions"""

    def handle(self, request, explanation_type):
        MAX_POINTS = 2_000 # tweak to hit gRPC size; lower = smaller message

        if explanation_type != 'featureExplanation':
            # we only support local feature explanations here
            raise ValueError(f"Unsupported explanation_type {explanation_type}")
        
        def big_csv_reduce(file_name, f, initial_value, chunk_size=100_000):
            result = initial_value
            for chunk in pd.read_csv(file_name, chunksize=chunk_size):
                result = f(result, chunk)
            return result

        def find_unique_values_big_csv(file_name, column_name):
            return big_csv_reduce(file_name, lambda result, chunk: result.union(chunk[column_name].unique()), set())
        
        def filter_csv_couple_chunked(file1, file2, column_name, column_value, chunk_size=100_000):
            reader1 = pd.read_csv(file1, chunksize=chunk_size)
            reader2 = pd.read_csv(file2, chunksize=chunk_size)
            filtered1 = []
            filtered2 = []
            for chunk1, chunk2 in zip(reader1, reader2):
                mask = chunk1[column_name] == column_value
                filtered1.append(chunk1[mask])
                filtered2.append(chunk2[mask])
            
            df1 = pd.concat(filtered1, ignore_index=True)
            df2 = pd.concat(filtered2, ignore_index=True)
            return df1, df2
        
        def filter_csv_triple_chunked(file1, file2, file3, column_name, column_value, chunk_size=100_000):
            reader1 = pd.read_csv(file1, chunksize=chunk_size)
            reader2 = pd.read_csv(file2, chunksize=chunk_size)
            reader3 = pd.read_csv(file3, chunksize=chunk_size)
            filtered1 = []
            filtered2 = []
            filtered3 = []
            for chunk1, chunk2, chunk3 in zip(reader1, reader2, reader3):
                mask = chunk1[column_name] == column_value
                filtered1.append(chunk1[mask])
                filtered2.append(chunk2[mask])
                filtered3.append(chunk3[mask])
            df1 = pd.concat(filtered1, ignore_index=True)
            df2 = pd.concat(filtered2, ignore_index=True)
            df3 = pd.concat(filtered3, ignore_index=True)
            return df1, df2, df3
        
        if request.HasField("segmentation_process_step"):
            segmentation_process_step = request.segmentation_process_step
        else:
            segmentation_process_step = None
        
        if segmentation_process_step == "return_available_indices":
            available_indices = find_unique_values_big_csv(request.data.X_test, 'instance_id')
            return xai_service_pb2.ExplanationsResponse(
                explainability_type = explanation_type,
                explanation_method = 'segmentation',
                segmentation_process_step = segmentation_process_step,
                plot_name = 'Available Indices',
                plot_descr = "Available Indices are the indices of the instances in the test set.",
                plot_type = 'List',
                available_indices = available_indices,
            )
        
        if segmentation_process_step == "return_instance":
            available_indices = find_unique_values_big_csv(request.data.X_test, 'instance_id')
            if request.HasField("instance_index"):
                instance_index = request.instance_index
            else:
                instance_index = list(available_indices)[0]

            test_instance_X, test_instance_Y, test_instance_preds = filter_csv_triple_chunked(
                file1=request.data.X_test,
                file2=request.data.Y_test,
                file3=request.data.Y_pred,
                column_name='instance_id',
                column_value=instance_index,
            )
            test_df = pd.concat([test_instance_X, test_instance_Y, test_instance_preds], axis="columns")
            instances, x_coords, y_coords, labels, predictions = df_to_instances(test_df)
            assert len(instances) == 1, f"Expected exactly one instance after filtering by instance_index, got {len(instances)}"
            assert list(instances.keys())[0] == instance_index, "Filtered instance index does not match requested index"
            instance = instances[instance_index]
            x_coords, y_coords = x_coords[instance_index], y_coords[instance_index]
            label, prediction = labels[instance_index], predictions[instance_index]
            mask = (instance[[0],1,:,:] == 1).astype(np.float32)
            query, x_coords, y_coords, gt, mask = instance, x_coords, y_coords, label, mask

            df_feats, _ = spatiotemporal_to_filtered_long_df(
                array=instance.squeeze(),  # (T, C, H, W)
                x_coords=np.asarray(x_coords),   # (H, W)
                y_coords=np.asarray(y_coords),   # (H, W)
                mask=mask.squeeze(),  # ensure (H,W) or (1,H,W)
                channel_names=['DEM', 'Mask', 'WD_IN', 'RAIN'],
                max_rows=MAX_POINTS,
            )
            labels_and_predictions = np.stack([
                gt.squeeze(),
                prediction.squeeze(),
            ], axis=0) # (2, H, W)
            df_targets, _ = spatial_multichannel_to_filtered_long_df(
                array=labels_and_predictions,
                x_coords=np.asarray(x_coords),   # (H, W)
                y_coords=np.asarray(y_coords),   # (H, W)
                mask=mask.squeeze(),  # ensure (H,W) or (1,H,W)
                channel_names=['GroundTruth', 'Prediction'],
                max_rows=MAX_POINTS,
            )

            table_contents_feats =  {
                col: xai_service_pb2.TableContents(
                    index=i+1,
                    values=df_feats[col].astype(str).tolist(),
                )
                for i, col in enumerate(df_feats.columns)
            }

            table_contents_targets = {
                col: xai_service_pb2.TableContents(
                    index=i+1,
                    values=df_targets[col].astype(str).tolist(),
                )
                for i, col in enumerate(df_targets.columns)
            }

            return xai_service_pb2.ExplanationsResponse(
                explainability_type  = explanation_type,
                explanation_method   = 'segmentation',
                explainability_model = request.model[0],
                plot_name            = 'Attributions',
                plot_descr           = (
                    "This method attributes the model's output to each input feature (pixel)."
                ),
                plot_type            = 'Table',
                hyperparameter_list = [],
                features_table=table_contents_feats,
                targets_table=table_contents_targets,
                features_table_columns=df_feats.columns.tolist(),
                targets_table_columns=df_targets.columns.tolist(),
                available_indices=available_indices,
            )
        
        # Load data and metadata from disk, as efficiently as possible
        model_path = request.model
        available_indices = find_unique_values_big_csv(request.data.X_test, 'instance_id')
        if request.HasField("instance_index"):
            instance_index = request.instance_index
        else:
            instance_index = list(available_indices)[0]
        test_instance_X, test_instance_Y, test_instance_preds = filter_csv_triple_chunked(
            file1=request.data.X_test,
            file2=request.data.Y_test,
            file3=request.data.Y_pred,
            column_name='instance_id',
            column_value=instance_index,
        )
        logger.debug(f"{test_instance_X.shape=}")
        logger.debug(f"{test_instance_Y.shape}")
        logger.debug(f"{test_instance_preds.shape}")

        test_df = pd.concat([test_instance_X, test_instance_Y, test_instance_preds], axis="columns")

        # query, x_coords, y_coords, gt, mask = parse_instance_from_request(request=request)
        # #logger.info(f"Parsed query with shape {query.shape}, mask {mask.shape}, gt {gt.shape}")
        # #logger.info(f"Parsed x_coords with shape {x_coords.shape}, y_coords {y_coords.shape}")
        print(test_df.shape)
        instances, x_coords, y_coords, labels, predictions = df_to_instances(test_df)
        assert len(instances) == 1, f"Expected exactly one instance after filtering by instance_index, got {len(instances)}"
        assert list(instances.keys())[0] == instance_index, "Filtered instance index does not match requested index"
        instance = instances[instance_index]
        x_coords, y_coords = x_coords[instance_index], y_coords[instance_index]
        label, prediction = labels[instance_index], predictions[instance_index]
        mask = (instance[[0],1,:,:] == 1).astype(np.float32)
        query, x_coords, y_coords, gt, mask = instance, x_coords, y_coords, label, mask

        # 2) Load model & device
        model, _ = _load_model(model_path[0])
        if not isinstance(model, torch.nn.Module):
            raise ValueError(f"Model {model_path[0]} is not a supported torch model")
        device = "cuda" if torch.cuda.is_available() else "cpu"  # TODO: however you track it
        model.to(device).eval()

        # 3) Prepare inputs: a single-instance batch
        #    assume dataset is an np.ndarray of shape [N,12,4,256,256]
        # inp = dataset[instance_index].unsqueeze(0).to(device)

        # test_input is a tensor of shape [1, T, C, H, W] (e.g., [1, 12, 4, 256, 256])
        test_input = treat_input(query, device=device)
        test_mask = treat_input(mask, device=device)
        test_ground_truth = treat_input(gt, device=device)

        logger.info(f"{test_input.shape=}")
        logger.info(f"{test_mask.shape=}")
        logger.info(f"{test_ground_truth.shape=}")

        baseline = torch.zeros_like(test_input)

        ig = IntegratedGradients(lambda inp: model_wrapper(inp, model=model, mask=test_mask))

        logger.info(f"Now computing attributions for {test_input.shape} input")
        attributions, delta = ig.attribute(
            test_input,
            baseline,
            target=None,
            n_steps=5,
            return_convergence_delta=True,
            internal_batch_size=1,
        )
        logger.info(f"Attributions computed with delta={delta}")
        logger.info(f"{attributions.shape=}")

        attributions_np = attributions.squeeze().detach().cpu().numpy()

        # compress to fit grpc message limits
        MAX_POINTS = 2_000   # tweak to hit gRPC size; lower = smaller message
        roi_mask = None      # optional: pass ROI here if you want those preserved

        df_attrs, _ = spatiotemporal_to_filtered_long_df(
            array=attributions_np,    # (T, C, H, W) or (1, T, C, H, W)
            x_coords=np.asarray(x_coords),   # (H, W)
            y_coords=np.asarray(y_coords),   # (H, W)
            mask=test_mask.detach().cpu().numpy().squeeze(),  # ensure (H,W) or (1,H,W)
            channel_names=['DEM', 'Mask', 'WD_IN', 'RAIN'],
            max_rows=MAX_POINTS,
        )
        df_feats, _ = spatiotemporal_to_filtered_long_df(
            array=test_input.squeeze().detach().cpu().numpy(),  # (T, C, H, W)
            x_coords=np.asarray(x_coords),   # (H, W)
            y_coords=np.asarray(y_coords),   # (H, W)
            mask=test_mask.detach().cpu().numpy().squeeze(),  # ensure (H,W) or (1,H,W)
            channel_names=['DEM', 'Mask', 'WD_IN', 'RAIN'],
            max_rows=MAX_POINTS,
        )
        labels_and_predictions = np.stack([
            gt.squeeze(),
            prediction.squeeze(),
        ], axis=0) # (2, H, W)
        df_targets, _ = spatial_multichannel_to_filtered_long_df(
            array=labels_and_predictions,
            x_coords=np.asarray(x_coords),   # (H, W)
            y_coords=np.asarray(y_coords),   # (H, W)
            mask=test_mask.detach().cpu().numpy().squeeze(),  # ensure (H,W) or (1,H,W)
            channel_names=['GroundTruth', 'Prediction'],
            max_rows=MAX_POINTS,
        )

        table_contents_feats =  {
            col: xai_service_pb2.TableContents(
                index=i+1,
                values=df_feats[col].astype(str).tolist(),
            )
            for i, col in enumerate(df_feats.columns)
        }

        table_contents_attrs =  {
            col: xai_service_pb2.TableContents(
                index=i+1,
                values=df_attrs[col].astype(str).tolist(),
            )
            for i, col in enumerate(df_attrs.columns)
        }

        table_contents_targets = {
            col: xai_service_pb2.TableContents(
                index=i+1,
                values=df_targets[col].astype(str).tolist(),
            )
            for i, col in enumerate(df_targets.columns)
        }

        return xai_service_pb2.ExplanationsResponse(
            explainability_type  = explanation_type,
            explanation_method   = 'segmentation',
            explainability_model = model_path[0],
            plot_name            = 'Attributions',
            plot_descr           = (
                "This method attributes the model's output to each input feature (pixel)."
            ),
            plot_type            = 'Table',
            hyperparameter_list = [],
            features_table=table_contents_feats,
            attributions_table=table_contents_attrs,
            targets_table=table_contents_targets,
            features_table_columns=df_feats.columns.tolist(),
            attributions_table_columns=df_attrs.columns.tolist(),
            targets_table_columns=df_targets.columns.tolist(),
            available_indices=available_indices,
        )

class SHAPHandler(BaseExplanationHandler):

    def handle(self, request, explanation_type):
        import shap

        if explanation_type == 'featureExplanation':
            # model_id = request.model_id
            model_path = request.model

            train_data = _load_dataset(request.data.X_train)
            train_labels = _load_dataset(request.data.Y_train)  
            test_data = _load_dataset(request.data.X_test)
            test_labels = _load_dataset(request.data.Y_test)         
           
            model, name = _load_model(model_path[0])
            idx = request.instance_index
            

            explainer, X_for_expl, feat_names = make_explainer_any(model, train_data, test_data)
            ex = explainer(X_for_expl)  
            try:
                ex.feature_names = feat_names
            except Exception:
                pass
            shap_explanations = shap_waterfall_payload(ex, idx=idx, class_idx=None, top_k=10, include_rest=False )

            return xai_service_pb2.ExplanationsResponse(
                explainability_type = explanation_type,
                explanation_method = 'shap',
                explainability_model = model_path[0],
                plot_name = 'SHAP',
                plot_descr = "SHAP (SHapley Additive exPlanations) is a method to explain any model's predictions by assigning each feature a contribution to a specific prediction.",
                plot_type = 'Bar Plot',
                xAxis = xai_service_pb2.Axis(
                            axis_name="E[f(X)] and f(x)", 
                            axis_values=[str(shap_explanations['expected_value']), str(shap_explanations['prediction_value'])], 
                            axis_type='numerical'  
                ),
                shap_contributions = [
                    xai_service_pb2.ShapContributions(
                    feature_name=str(r["feature"]),
                    feature_value=float(r["feature_value"]) if r.get("feature_value") is not None else 0.0,
                    shap_value=float(r["shap"]),
                )
                for r in shap_explanations["contributions"]],
            )

class FeatureImportanceHandler(BaseExplanationHandler):

    def handle(self, request, explanation_type):

        if explanation_type == 'experimentExplanation':
            experiment_configs = request.experiment_configs

            logger.debug("List of experiment configs received for PDP:")
            logger.debug(f"{experiment_configs=}")
            keep_common_variability_points(experiment_configs)

            hyper_space = create_hyperspace(experiment_configs)
            hyper_df, sorted_metrics = create_hyper_df(experiment_configs)

            logger.info('Training Surrogate Model...')
            surrogate_model = self._load_or_train_surrogate_model(hyper_df, sorted_metrics)
            logger.info("Trained Surrogate Model.")

            logger.info("Computing Feature Importances using Permutation Importance...")
            result = permutation_importance(
                surrogate_model,
                hyper_df,
                np.array(sorted_metrics),
                scoring='neg_root_mean_squared_error',
                n_repeats=5,
                random_state=42
            )
            logger.info("Computed Feature Importances. Now formatting and returning response.")

            feature_importances = list(zip(hyper_df.columns, result.importances_mean))
            sorted_features = sorted(feature_importances, key=lambda x: x[1], reverse=True)
            
            feat_imp_df = pd.DataFrame(sorted_features, columns=['Feature', 'Importance'])

            table_contents_featimp = {
                col: xai_service_pb2.TableContents(
                    index=i+1,
                    values=feat_imp_df[col].astype(str).tolist(),
                )
                for i, col in enumerate(feat_imp_df.columns)
            }
            
            return xai_service_pb2.ExplanationsResponse(
                explainability_type=explanation_type,
                explanation_method='feature_importance',
                explainability_model="",
                plot_name='Experiment Variability Point Importance Plot',
                plot_descr="Experiment Variability Point Importance quantifies the impact of different options on the specified metric.",
                plot_type='Table',
                hyperparameter_list = feat_imp_df['Feature'].tolist(),
                table_contents=table_contents_featimp,
            )

        else:
            raise ValueError(f"Unknown explanation type: {explanation_type}")
