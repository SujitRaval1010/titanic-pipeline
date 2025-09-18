# smart_model_registry.py

import os
import json
import hashlib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from pyspark.sql import SparkSession

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class SmartModelRegistry:
    def __init__(self, model_name: str, catalog: str = None, schema: str = None, experiment_name: str = None):
        """
        Initialize Smart Model Registry that prevents duplicate parameter registrations
        
        Args:
            model_name: Name of the model (without catalog.schema prefix)
            catalog: Unity Catalog name (will auto-discover if None)
            schema: Schema name (defaults to 'default' or 'ml')
            experiment_name: MLflow experiment name
        """
        self.model_name_simple = model_name
        self.experiment_name = experiment_name or "/Shared/titanic"
        self.client = MlflowClient()
        
        # Set MLflow configuration for Unity Catalog
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")  # Unity Catalog registry
        
        # Auto-discover or create catalog and schema
        self.catalog, self.schema = self._setup_catalog_schema(catalog, schema)
        
        # Full model name with catalog and schema
        self.model_name = f"{self.catalog}.{self.schema}.{model_name}"
        
        # Cache for parameter-to-version mapping
        self._param_version_cache = None
        
        logging.info(f"Initialized Smart Model Registry: {self.model_name}")
    
    def _get_available_catalogs(self) -> List[str]:
        """Get list of available catalogs"""
        try:
            spark = SparkSession.builder.getOrCreate()
            catalogs_df = spark.sql("SHOW CATALOGS")
            return [row['catalog'] for row in catalogs_df.collect()]
        except Exception as e:
            logging.warning(f"Could not retrieve catalogs: {e}")
            return []
    
    def _get_available_schemas(self, catalog: str) -> List[str]:
        """Get list of available schemas in a catalog"""
        try:
            spark = SparkSession.builder.getOrCreate()
            schemas_df = spark.sql(f"SHOW SCHEMAS IN {catalog}")
            return [row['namespace'] for row in schemas_df.collect()]
        except Exception as e:
            logging.warning(f"Could not retrieve schemas for catalog {catalog}: {e}")
            return []
    
    def _create_catalog_if_needed(self, catalog_name: str) -> bool:
        """Try to create catalog if it doesn't exist"""
        try:
            spark = SparkSession.builder.getOrCreate()
            spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
            logging.info(f"✅ Catalog '{catalog_name}' created/verified")
            return True
        except Exception as e:
            logging.warning(f"Could not create catalog '{catalog_name}': {e}")
            return False
    
    def _create_schema_if_needed(self, catalog: str, schema: str) -> bool:
        """Try to create schema if it doesn't exist"""
        try:
            spark = SparkSession.builder.getOrCreate()
            spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
            logging.info(f"✅ Schema '{catalog}.{schema}' created/verified")
            return True
        except Exception as e:
            logging.warning(f"Could not create schema '{catalog}.{schema}': {e}")
            return False
    
    def _setup_catalog_schema(self, catalog: str = None, schema: str = None) -> Tuple[str, str]:
        """Setup or discover appropriate catalog and schema"""
        available_catalogs = self._get_available_catalogs()
        
        logging.info(f"Available catalogs: {available_catalogs}")
        
        # Determine catalog
        if catalog:
            target_catalog = catalog
        else:
            preferred_catalogs = ['main', 'hive_metastore', 'workspace']
            target_catalog = None
            
            for pref_catalog in preferred_catalogs:
                if pref_catalog in available_catalogs:
                    target_catalog = pref_catalog
                    logging.info(f"Using existing catalog: {target_catalog}")
                    break
            
            if not target_catalog:
                if self._create_catalog_if_needed('main'):
                    target_catalog = 'main'
                elif available_catalogs:
                    target_catalog = available_catalogs[0]
                    logging.info(f"Using first available catalog: {target_catalog}")
                else:
                    target_catalog = 'hive_metastore'
                    logging.info(f"Falling back to catalog: {target_catalog}")
        
        # Determine schema
        if schema:
            target_schema = schema
        else:
            available_schemas = self._get_available_schemas(target_catalog)
            logging.info(f"Available schemas in '{target_catalog}': {available_schemas}")
            
            preferred_schemas = ['ml', 'default', 'machine_learning']
            target_schema = None
            
            for pref_schema in preferred_schemas:
                if pref_schema in available_schemas:
                    target_schema = pref_schema
                    logging.info(f"Using existing schema: {target_schema}")
                    break
            
            if not target_schema:
                if self._create_schema_if_needed(target_catalog, 'ml'):
                    target_schema = 'ml'
                elif self._create_schema_if_needed(target_catalog, 'default'):
                    target_schema = 'default'
                else:
                    target_schema = available_schemas[0] if available_schemas else 'default'
        
        if not self._create_schema_if_needed(target_catalog, target_schema):
            logging.warning(f"Could not verify schema '{target_catalog}.{target_schema}'")
        
        logging.info(f"Final setup: catalog='{target_catalog}', schema='{target_schema}'")
        return target_catalog, target_schema
        
    def _calculate_parameter_signature(self, params: Dict) -> str:
        """
        Calculate a consistent signature based only on model parameters
        that affect the model's behavior (not run-specific params)
        """
        # Define parameters that actually affect model behavior
        model_relevant_params = [
            'algorithm', 'n_estimators', 'max_depth', 'min_samples_split', 
            'min_samples_leaf', 'random_state', 'bootstrap', 'criterion',
            'max_features', 'min_impurity_decrease', 'class_weight',
            'max_leaf_nodes', 'min_weight_fraction_leaf', 'oob_score',
            'warm_start', 'ccp_alpha'
        ]
        
        # Filter and normalize parameters
        filtered_params = {}
        for key, value in params.items():
            if key in model_relevant_params:
                # Normalize None values and convert to string for consistent comparison
                if value is None or value == 'None':
                    filtered_params[key] = 'None'
                else:
                    filtered_params[key] = str(value)
        
        # Create consistent signature
        param_string = json.dumps(sorted(filtered_params.items()), sort_keys=True)
        signature = hashlib.md5(param_string.encode()).hexdigest()
        
        logging.info(f"Parameter signature: {signature}")
        logging.info(f"Relevant parameters: {filtered_params}")
        
        return signature
    
    def _build_parameter_version_mapping(self) -> Dict[str, Dict]:
        """
        Build a mapping of parameter signatures to their corresponding versions
        
        Returns:
            Dict mapping signature -> {version, run_id, params, metrics, accuracy}
        """
        if self._param_version_cache is not None:
            return self._param_version_cache
        
        try:
            # Get all model versions
            model_versions = self.client.search_model_versions(f"name='{self.model_name}'")
            
            param_mapping = {}
            
            for model_version in model_versions:
                try:
                    # Get run details
                    run = self.client.get_run(model_version.run_id)
                    params = run.data.params
                    metrics = run.data.metrics
                    
                    # Calculate parameter signature
                    param_signature = self._calculate_parameter_signature(params)
                    
                    # Store mapping (keep the highest performing version for each signature)
                    if param_signature not in param_mapping:
                        param_mapping[param_signature] = {
                            'version': model_version.version,
                            'run_id': model_version.run_id,
                            'params': params,
                            'metrics': metrics,
                            'accuracy': float(metrics.get('accuracy', 0)),
                            'creation_timestamp': model_version.creation_timestamp
                        }
                    else:
                        # If we have the same parameters, keep the better performing version
                        existing_accuracy = param_mapping[param_signature]['accuracy']
                        current_accuracy = float(metrics.get('accuracy', 0))
                        
                        if current_accuracy > existing_accuracy:
                            param_mapping[param_signature] = {
                                'version': model_version.version,
                                'run_id': model_version.run_id,
                                'params': params,
                                'metrics': metrics,
                                'accuracy': current_accuracy,
                                'creation_timestamp': model_version.creation_timestamp
                            }
                            logging.info(f"Updated mapping for signature {param_signature} to version {model_version.version} (better accuracy: {current_accuracy})")
                
                except Exception as e:
                    logging.warning(f"Could not process model version {model_version.version}: {e}")
                    continue
            
            self._param_version_cache = param_mapping
            logging.info(f"Built parameter mapping for {len(param_mapping)} unique parameter combinations")
            
            return param_mapping
            
        except MlflowException as e:
            if "RESOURCE_DOES_NOT_EXIST" in str(e) or "does not exist" in str(e).lower():
                logging.info(f"Model '{self.model_name}' not found in Unity Catalog. Starting fresh.")
                return {}
            else:
                logging.error(f"Error building parameter mapping: {e}")
                raise e
    
    def register_model_if_needed(self, run_id: str, model_path: str = "model") -> Dict[str, Any]:
        """
        Smart model registration that reuses existing versions for identical parameters
        
        Args:
            run_id: MLflow run ID containing the model
            model_path: Path to model within the run (default: "model")
            
        Returns:
            Dict with registration result
        """
        try:
            # Get current run details
            run = self.client.get_run(run_id)
            current_params = run.data.params
            current_metrics = run.data.metrics
            current_accuracy = float(current_metrics.get('accuracy', 0))
            
            logging.info(f"Evaluating model registration for run: {run_id}")
            logging.info(f"Current accuracy: {current_accuracy}")
            
            # Calculate parameter signature for current run
            current_signature = self._calculate_parameter_signature(current_params)
            
            # Build parameter-version mapping
            param_mapping = self._build_parameter_version_mapping()
            
            # Check if we already have a version with these exact parameters
            if current_signature in param_mapping:
                existing_version_info = param_mapping[current_signature]
                existing_version = existing_version_info['version']
                existing_accuracy = existing_version_info['accuracy']
                existing_run_id = existing_version_info['run_id']
                
                logging.info(f"Found existing version {existing_version} with same parameters")
                logging.info(f"Existing accuracy: {existing_accuracy}, Current accuracy: {current_accuracy}")
                
                # Check if current run is significantly better
                improvement_threshold = float(os.getenv("MODEL_ACCURACY_THRESHOLD", "0.01"))
                accuracy_improvement = current_accuracy - existing_accuracy
                
                if accuracy_improvement > improvement_threshold:
                    # Register new version because of significant improvement
                    logging.info(f"Registering new version due to significant improvement: +{accuracy_improvement:.4f}")
                    
                    model_uri = f"runs:/{run_id}/{model_path}"
                    model_version = mlflow.register_model(
                        model_uri=model_uri,
                        name=self.model_name,
                        tags={
                            "algorithm": current_params.get("algorithm", "RandomForest"),
                            "n_estimators": current_params.get("n_estimators", "100"),
                            "accuracy": str(current_accuracy),
                            "registration_reason": f"Improved accuracy: {existing_accuracy:.4f} -> {current_accuracy:.4f}",
                            "replaces_version": str(existing_version),
                            "parameter_signature": current_signature
                        }
                    )
                    
                    # Invalidate cache
                    self._param_version_cache = None
                    
                    return {
                        "registered": True,
                        "version": model_version.version,
                        "reason": f"Improved accuracy over version {existing_version}: {existing_accuracy:.4f} -> {current_accuracy:.4f} (+{accuracy_improvement:.4f})",
                        "model_uri": model_uri,
                        "model_name": self.model_name,
                        "run_id": run_id,
                        "replaced_version": existing_version
                    }
                
                else:
                    # Same parameters, no significant improvement - DO NOT register
                    logging.info(f"Skipping registration - version {existing_version} already exists with same/better parameters")
                    
                    return {
                        "registered": False,
                        "reason": f"Parameters identical to existing version {existing_version}. Accuracy difference: {accuracy_improvement:+.4f} (threshold: {improvement_threshold})",
                        "existing_version": existing_version,
                        "existing_accuracy": existing_accuracy,
                        "current_accuracy": current_accuracy,
                        "existing_run_id": existing_run_id,
                        "current_run_id": run_id,
                        "model_name": self.model_name,
                        "parameter_signature": current_signature
                    }
            
            else:
                # New parameter combination - register new version
                logging.info("New parameter combination detected - registering new version")
                
                model_uri = f"runs:/{run_id}/{model_path}"
                model_version = mlflow.register_model(
                    model_uri=model_uri,
                    name=self.model_name,
                    tags={
                        "algorithm": current_params.get("algorithm", "RandomForest"),
                        "n_estimators": current_params.get("n_estimators", "100"),
                        "accuracy": str(current_accuracy),
                        "registration_reason": "New parameter combination",
                        "parameter_signature": current_signature
                    }
                )
                
                # Invalidate cache
                self._param_version_cache = None
                
                return {
                    "registered": True,
                    "version": model_version.version,
                    "reason": "New parameter combination",
                    "model_uri": model_uri,
                    "model_name": self.model_name,
                    "run_id": run_id,
                    "parameter_signature": current_signature
                }
                
        except Exception as e:
            logging.error(f"Error in smart model registration: {e}")
            raise e
    
    def set_model_alias(self, version: str, alias: str) -> None:
        """Set an alias for a model version"""
        try:
            self.client.set_registered_model_alias(
                name=self.model_name,
                alias=alias,
                version=version
            )
            logging.info(f"✅ Model version {version} set with alias '{alias}'")
        except Exception as e:
            logging.error(f"Error setting model alias '{alias}': {e}")
            raise e
    
    def get_parameter_mapping_info(self) -> Dict:
        """Get information about parameter-to-version mapping"""
        try:
            param_mapping = self._build_parameter_version_mapping()
            
            mapping_info = []
            for signature, info in param_mapping.items():
                # Get key parameters for display
                params = info['params']
                key_params = {
                    'algorithm': params.get('algorithm', 'N/A'),
                    'n_estimators': params.get('n_estimators', 'N/A'),
                    'max_depth': params.get('max_depth', 'N/A'),
                    'min_samples_split': params.get('min_samples_split', 'N/A')
                }
                
                mapping_info.append({
                    'signature': signature,
                    'version': info['version'],
                    'accuracy': info['accuracy'],
                    'key_parameters': key_params,
                    'run_id': info['run_id']
                })
            
            # Sort by version
            mapping_info.sort(key=lambda x: int(x['version']))
            
            return {
                "model_name": self.model_name,
                "total_parameter_combinations": len(mapping_info),
                "parameter_mappings": mapping_info
            }
            
        except Exception as e:
            logging.error(f"Error getting parameter mapping info: {e}")
            return {"error": str(e)}


def main():
    """Main function to demonstrate smart model registration"""
    
    # Configuration
    MODEL_NAME = os.getenv("MODEL_NAME", "titanic_survival_model")
    CATALOG = os.getenv("UNITY_CATALOG_NAME")  # Auto-discover if None
    SCHEMA = os.getenv("UNITY_SCHEMA_NAME")    # Auto-discover if None
    EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/titanic")
    
    print(f"🧠 Smart Model Registry")
    print(f"   Model Name: {MODEL_NAME}")
    print(f"   Catalog: {CATALOG or 'auto-discover'}")
    print(f"   Schema: {SCHEMA or 'auto-discover'}")
    
    # Initialize registry
    registry = SmartModelRegistry(MODEL_NAME, CATALOG, SCHEMA, EXPERIMENT_NAME)
    
    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        
        if experiment is None:
            logging.error(f"Experiment '{EXPERIMENT_NAME}' not found!")
            return
        
        # Get latest run
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if runs.empty:
            logging.error("No runs found in the experiment!")
            return
        
        latest_run_id = runs.iloc[0]['run_id']
        logging.info(f"Latest run ID: {latest_run_id}")
        
        # Try smart registration
        result = registry.register_model_if_needed(latest_run_id)
        
        print("\n" + "="*60)
        print("SMART MODEL REGISTRATION RESULT")
        print("="*60)
        print(f"Registered: {result['registered']}")
        print(f"Reason: {result['reason']}")
        print(f"Model Name: {result['model_name']}")
        
        if result['registered']:
            print(f"New Version: {result['version']}")
            print(f"Model URI: {result['model_uri']}")
            if 'replaced_version' in result:
                print(f"Replaced Version: {result['replaced_version']}")
        else:
            print(f"Existing Version: {result['existing_version']}")
            print(f"Existing Accuracy: {result['existing_accuracy']}")
            print(f"Current Accuracy: {result['current_accuracy']}")
        
        # Show parameter mapping
        print("\n" + "="*60)
        print("PARAMETER-TO-VERSION MAPPING")
        print("="*60)
        
        mapping_info = registry.get_parameter_mapping_info()
        if "error" not in mapping_info:
            print(f"Total Parameter Combinations: {mapping_info['total_parameter_combinations']}")
            print("\nParameter Mappings:")
            
            for mapping in mapping_info['parameter_mappings']:
                print(f"  Version {mapping['version']} (Accuracy: {mapping['accuracy']:.4f}):")
                for param, value in mapping['key_parameters'].items():
                    print(f"    {param}: {value}")
                print(f"    Signature: {mapping['signature'][:16]}...")
                print()
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise e


if __name__ == "__main__":
    main()