# model_registry.py

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

class UnityCatalogModelRegistry:
    def __init__(self, model_name: str, catalog: str = None, schema: str = None, experiment_name: str = None):
        """
        Initialize Unity Catalog Model Registry with intelligent versioning
        
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
        
        logging.info(f"Initialized Unity Catalog Model Registry: {self.model_name}")
    
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
        """
        Setup or discover appropriate catalog and schema
        
        Returns:
            Tuple[str, str]: (catalog_name, schema_name)
        """
        available_catalogs = self._get_available_catalogs()
        
        logging.info(f"Available catalogs: {available_catalogs}")
        
        # Determine catalog
        if catalog:
            target_catalog = catalog
        else:
            # Try to find a suitable catalog
            preferred_catalogs = ['main', 'hive_metastore', 'workspace']
            target_catalog = None
            
            for pref_catalog in preferred_catalogs:
                if pref_catalog in available_catalogs:
                    target_catalog = pref_catalog
                    logging.info(f"Using existing catalog: {target_catalog}")
                    break
            
            if not target_catalog:
                # Try to create 'main' catalog
                if self._create_catalog_if_needed('main'):
                    target_catalog = 'main'
                elif available_catalogs:
                    # Use the first available catalog
                    target_catalog = available_catalogs[0]
                    logging.info(f"Using first available catalog: {target_catalog}")
                else:
                    # Fallback to hive_metastore (usually always available)
                    target_catalog = 'hive_metastore'
                    logging.info(f"Falling back to catalog: {target_catalog}")
        
        # Determine schema
        if schema:
            target_schema = schema
        else:
            # Try common schema names
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
                # Try to create 'ml' schema
                if self._create_schema_if_needed(target_catalog, 'ml'):
                    target_schema = 'ml'
                elif self._create_schema_if_needed(target_catalog, 'default'):
                    target_schema = 'default'
                else:
                    # Use first available schema or fallback
                    target_schema = available_schemas[0] if available_schemas else 'default'
        
        # Verify the final setup
        if not self._create_schema_if_needed(target_catalog, target_schema):
            logging.warning(f"Could not verify schema '{target_catalog}.{target_schema}'")
        
        logging.info(f"Final setup: catalog='{target_catalog}', schema='{target_schema}'")
        return target_catalog, target_schema
        
    def _calculate_model_signature(self, run_id: str, params: Dict, metrics: Dict) -> str:
        """
        Calculate a unique signature for the model based on:
        - Model parameters (excluding run-specific params)
        - Algorithm type
        - Feature set (if available)
        """
        try:
            # Filter out run-specific parameters that don't affect model behavior
            filtered_params = {}
            model_relevant_params = [
                'algorithm', 'n_estimators', 'max_depth', 'min_samples_split', 
                'min_samples_leaf', 'random_state', 'bootstrap', 'criterion',
                'max_features', 'min_impurity_decrease', 'class_weight'
            ]
            
            for key, value in params.items():
                if key in model_relevant_params:
                    # Convert None to string for consistent comparison
                    filtered_params[key] = str(value) if value is not None else 'None'
            
            # Create signature components
            signature_data = {
                "algorithm": filtered_params.get("algorithm", "RandomForest"),
                "params": sorted(filtered_params.items()),
            }
            
            # Try to get feature information from the model
            try:
                model_info = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
                if hasattr(model_info, 'feature_names_in_'):
                    signature_data["features"] = sorted(model_info.feature_names_in_.tolist())
                elif hasattr(model_info, 'n_features_in_'):
                    signature_data["n_features"] = model_info.n_features_in_
                
                # Add model-specific attributes
                if hasattr(model_info, 'n_estimators'):
                    signature_data["n_estimators"] = model_info.n_estimators
                if hasattr(model_info, 'max_depth'):
                    signature_data["max_depth"] = str(model_info.max_depth)
                    
            except Exception as model_error:
                logging.warning(f"Could not extract model information: {model_error}")
            
            # Create hash of signature
            signature_str = json.dumps(signature_data, sort_keys=True)
            signature_hash = hashlib.md5(signature_str.encode()).hexdigest()
            
            logging.info(f"Model signature calculated: {signature_hash}")
            logging.info(f"Signature components: {signature_data}")
            
            return signature_hash
            
        except Exception as e:
            logging.error(f"Error calculating model signature: {e}")
            # Fallback to basic param hash
            basic_params = {k: v for k, v in params.items() if k in ['algorithm', 'n_estimators', 'max_depth']}
            return hashlib.md5(str(sorted(basic_params.items())).encode()).hexdigest()
    
    def _get_latest_registered_model_info(self) -> Optional[Dict]:
        """Get information about the latest registered model version in Unity Catalog"""
        try:
            # Search for model versions using Unity Catalog
            model_versions = self.client.search_model_versions(f"name='{self.model_name}'")
            
            if not model_versions:
                logging.info(f"No versions found for model '{self.model_name}'")
                return None
            
            # Get the most recent version by version number
            latest_version = max(model_versions, key=lambda x: int(x.version))
            
            # Get the run details for this version
            run = self.client.get_run(latest_version.run_id)
            
            return {
                "version": latest_version.version,
                "run_id": latest_version.run_id,
                "params": run.data.params,
                "metrics": run.data.metrics,
                "signature": self._calculate_model_signature(
                    latest_version.run_id, 
                    run.data.params, 
                    run.data.metrics
                )
            }
            
        except MlflowException as e:
            if "RESOURCE_DOES_NOT_EXIST" in str(e) or "does not exist" in str(e).lower():
                logging.info(f"Model '{self.model_name}' not found in Unity Catalog. Will create new model.")
                return None
            else:
                logging.error(f"Error getting latest model info: {e}")
                raise e
    
    def _should_register_new_version(self, 
                                   current_params: Dict, 
                                   current_metrics: Dict, 
                                   current_run_id: str) -> Tuple[bool, str]:
        """
        Determine if a new model version should be registered
        
        Returns:
            Tuple[bool, str]: (should_register, reason)
        """
        latest_model_info = self._get_latest_registered_model_info()
        
        # If no model exists, register the first version
        if latest_model_info is None:
            return True, "First model version"
        
        # Calculate current model signature
        current_signature = self._calculate_model_signature(
            current_run_id, current_params, current_metrics
        )
        
        logging.info(f"Current model signature: {current_signature}")
        logging.info(f"Previous model signature: {latest_model_info['signature']}")
        
        # Compare signatures - this is the primary check
        if current_signature != latest_model_info["signature"]:
            return True, "Model signature changed (parameters, algorithm, or features different)"
        
        # If signatures match, check for significant performance improvement
        accuracy_threshold = float(os.getenv("MODEL_ACCURACY_THRESHOLD", "0.01"))  # 1% improvement
        
        current_accuracy = float(current_metrics.get("accuracy", 0))
        previous_accuracy = float(latest_model_info["metrics"].get("accuracy", 0))
        
        accuracy_diff = current_accuracy - previous_accuracy
        
        logging.info(f"Accuracy comparison: current={current_accuracy:.4f}, previous={previous_accuracy:.4f}, diff={accuracy_diff:.4f}, threshold={accuracy_threshold}")
        
        if accuracy_diff > accuracy_threshold:
            return True, f"Significant accuracy improvement: {previous_accuracy:.4f} -> {current_accuracy:.4f} (improvement: +{accuracy_diff:.4f})"
        
        # Check if forced registration is requested
        if os.getenv("FORCE_MODEL_REGISTRATION", "false").lower() == "true":
            return True, "Forced registration via environment variable"
        
        # Check if the run IDs are different but everything else is the same
        if current_run_id == latest_model_info["run_id"]:
            return False, "Identical run - model already registered"
        
        return False, (
            f"No significant changes detected. "
            f"Latest version: {latest_model_info['version']}, "
            f"Accuracy diff: {accuracy_diff:+.4f} (threshold: {accuracy_threshold}), "
            f"Signature match: ✓"
        )
    
    def register_model_if_needed(self, run_id: str, model_path: str = "model") -> Dict[str, Any]:
        """
        Register model in Unity Catalog only if it's different from the latest version
        
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
            
            logging.info(f"Evaluating model registration for run: {run_id}")
            
            # Check if we should register a new version
            should_register, reason = self._should_register_new_version(
                current_params, current_metrics, run_id
            )
            
            if should_register:
                logging.info(f"Registering new model version. Reason: {reason}")
                
                # Register the model in Unity Catalog
                model_uri = f"runs:/{run_id}/{model_path}"
                
                try:
                    model_version = mlflow.register_model(
                        model_uri=model_uri,
                        name=self.model_name,
                        tags={
                            "algorithm": current_params.get("algorithm", "RandomForest"),
                            "n_estimators": current_params.get("n_estimators", "100"),
                            "accuracy": str(current_metrics.get("accuracy", 0)),
                            "registration_reason": reason,
                            "catalog": self.catalog,
                            "schema": self.schema
                        }
                    )
                    
                    logging.info(f"✅ Model registered as version {model_version.version} in Unity Catalog")
                    
                    return {
                        "registered": True,
                        "version": model_version.version,
                        "reason": reason,
                        "model_uri": model_uri,
                        "model_name": self.model_name,
                        "run_id": run_id
                    }
                    
                except Exception as reg_error:
                    logging.error(f"Error registering model in Unity Catalog: {reg_error}")
                    # Provide specific troubleshooting
                    error_msg = str(reg_error)
                    if "CATALOG_DOES_NOT_EXIST" in error_msg:
                        logging.error(f"❌ Catalog '{self.catalog}' does not exist")
                        logging.error("💡 Try running: spark.sql('CREATE CATALOG IF NOT EXISTS {self.catalog}')")
                    elif "SCHEMA_DOES_NOT_EXIST" in error_msg:
                        logging.error(f"❌ Schema '{self.catalog}.{self.schema}' does not exist")
                        logging.error(f"💡 Try running: spark.sql('CREATE SCHEMA IF NOT EXISTS {self.catalog}.{self.schema}')")
                    elif "PERMISSION_DENIED" in error_msg:
                        logging.error("❌ Permission denied - check Unity Catalog permissions")
                    
                    raise reg_error
                    
            else:
                logging.info(f"⏭️  Skipping registration. Reason: {reason}")
                
                return {
                    "registered": False,
                    "reason": reason,
                    "latest_version": self._get_latest_registered_model_info()["version"] if self._get_latest_registered_model_info() else "None",
                    "model_name": self.model_name,
                    "run_id": run_id
                }
                
        except Exception as e:
            logging.error(f"Error in model registration: {e}")
            raise e
    
    def set_model_alias(self, version: str, alias: str) -> None:
        """
        Set an alias for a model version (Unity Catalog uses aliases instead of stages)
        
        Args:
            version: Model version to set alias for
            alias: Alias name (e.g., "champion", "challenger", "staging")
        """
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
    
    def get_model_info(self) -> Dict:
        """Get comprehensive information about registered models in Unity Catalog"""
        try:
            model_versions = self.client.search_model_versions(f"name='{self.model_name}'")
            
            if not model_versions:
                return {"message": f"No versions found for model '{self.model_name}' in Unity Catalog"}
            
            versions_info = []
            for version in model_versions:
                run = self.client.get_run(version.run_id)
                
                # Get aliases for this version
                try:
                    model_details = self.client.get_registered_model(self.model_name)
                    aliases = [alias.alias for alias in model_details.aliases if alias.version == version.version]
                except:
                    aliases = []
                
                versions_info.append({
                    "version": version.version,
                    "aliases": aliases,
                    "run_id": version.run_id,
                    "accuracy": run.data.metrics.get("accuracy", "N/A"),
                    "n_estimators": run.data.params.get("n_estimators", "N/A"),
                    "algorithm": run.data.params.get("algorithm", "N/A"),
                    "creation_timestamp": version.creation_timestamp,
                    "tags": version.tags
                })
            
            # Sort by version number
            versions_info.sort(key=lambda x: int(x["version"]), reverse=True)
            
            return {
                "model_name": self.model_name,
                "catalog": self.catalog,
                "schema": self.schema,
                "total_versions": len(versions_info),
                "versions": versions_info
            }
            
        except Exception as e:
            logging.error(f"Error getting model info: {e}")
            return {"error": str(e)}


def show_catalog_info():
    """Helper function to show available catalogs and schemas"""
    try:
        spark = SparkSession.builder.getOrCreate()
        
        print("🏗️  UNITY CATALOG ENVIRONMENT")
        print("="*50)
        
        # Show catalogs
        catalogs_df = spark.sql("SHOW CATALOGS")
        catalogs = [row['catalog'] for row in catalogs_df.collect()]
        print(f"📚 Available Catalogs: {catalogs}")
        
        # Show schemas for each catalog
        for catalog in catalogs:
            try:
                schemas_df = spark.sql(f"SHOW SCHEMAS IN {catalog}")
                schemas = [row['namespace'] for row in schemas_df.collect()]
                print(f"📁 Schemas in '{catalog}': {schemas}")
            except Exception as e:
                print(f"❌ Could not access schemas in '{catalog}': {e}")
        
        return catalogs
        
    except Exception as e:
        print(f"❌ Error accessing Unity Catalog: {e}")
        return []


def main():
    """Main function to demonstrate usage with Unity Catalog"""
    
    # Show environment info first
    available_catalogs = show_catalog_info()
    
    if not available_catalogs:
        print("\n❌ No Unity Catalog access detected!")
        print("Please ensure Unity Catalog is enabled and you have proper permissions.")
        return
    
    # Configuration - Updated for Unity Catalog
    MODEL_NAME = os.getenv("MODEL_NAME", "titanic_survival_model")
    CATALOG = os.getenv("UNITY_CATALOG_NAME")  # Let auto-discovery handle if None
    SCHEMA = os.getenv("UNITY_SCHEMA_NAME")   # Let auto-discovery handle if None
    EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/titanic")
    
    print(f"\n🤖 Model Configuration:")
    print(f"   Model Name: {MODEL_NAME}")
    print(f"   Catalog: {CATALOG or 'auto-discover'}")
    print(f"   Schema: {SCHEMA or 'auto-discover'}")
    
    # Initialize registry (will auto-discover catalog/schema if not provided)
    registry = UnityCatalogModelRegistry(MODEL_NAME, CATALOG, SCHEMA, EXPERIMENT_NAME)
    
    # Get the latest run from the experiment
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
        
        # Try to register the model
        result = registry.register_model_if_needed(latest_run_id)
        
        print("\n" + "="*50)
        print("MODEL REGISTRATION RESULT")
        print("="*50)
        print(f"Registered: {result['registered']}")
        print(f"Reason: {result['reason']}")
        print(f"Model Name: {result['model_name']}")
        
        if result['registered']:
            print(f"New Version: {result['version']}")
            print(f"Model URI: {result['model_uri']}")
            
            # Set alias if it's a significant improvement
            if "accuracy improvement" in result['reason'].lower():
                registry.set_model_alias(str(result['version']), "challenger")
                print(f"🏆 Model version {result['version']} set as 'challenger'")
        else:
            print(f"Latest Version: {result.get('latest_version', 'N/A')}")
        
        # Show model information
        print("\n" + "="*50)
        print("UNITY CATALOG MODEL REGISTRY STATUS")
        print("="*50)
        model_info = registry.get_model_info()
        
        if "error" not in model_info:
            print(f"Model: {model_info['model_name']}")
            print(f"Catalog: {model_info['catalog']}")
            print(f"Schema: {model_info['schema']}")
            print(f"Total Versions: {model_info['total_versions']}")
            print("\nVersion History:")
            for version in model_info['versions'][:5]:  # Show latest 5 versions
                aliases_str = f" [{', '.join(version['aliases'])}]" if version['aliases'] else ""
                print(f"  v{version['version']}{aliases_str} - "
                      f"Accuracy: {version['accuracy']}, "
                      f"Algorithm: {version['algorithm']}, "
                      f"n_estimators: {version['n_estimators']}")
        else:
            print(f"Error: {model_info['error']}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        print(f"\n❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check Unity Catalog permissions")
        print("2. Verify catalog and schema exist")
        print("3. Ensure experiment has runs with trained models")
        raise e


if __name__ == "__main__":
    main()