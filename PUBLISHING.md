# Publishing Guide for fg-strategy-modelling-core

This document describes all the changes made to transform the repository into a professional, publishable Python package, and provides instructions for publishing to PyPI.

## Changes Made

### 1. Package Configuration Files

#### ✅ `pyproject.toml` - Modern Python Packaging
- **Purpose**: Modern Python packaging configuration (PEP 517)
- **Features**:
  - Complete project metadata
  - Dependencies and optional dependencies (dev, aws, gcs, cloudflare)
  - Tool configurations (black, isort, mypy, pytest)
  - Build system specification

#### ✅ `setup.py` - Backward Compatibility
- **Purpose**: Backward compatibility for older installation methods
- **Uses**: Reads from pyproject.toml for configuration

#### ✅ `MANIFEST.in` - Distribution Files
- **Purpose**: Includes additional files in source distributions
- **Includes**: README, LICENSE, requirements, and Python source files

### 2. Comprehensive Documentation

#### ✅ `README.md` - Complete Usage Guide (Updated)
**Sections added:**
- Feature highlights
- Installation instructions (PyPI and from source)
- Quick start with 4 practical examples
- Full environment variables reference with 30+ variables documented
- Detailed API reference for all main classes and functions
- Configuration guide for all supported storage backends
- Troubleshooting guide
- Contributing guidelines
- License information

#### ✅ `.env.example` (Created)
- Example environment configuration template
- All supported variables with descriptions
- Ready to copy and customize

### 3. Environment Variable Validation

#### ✅ `src/env_validator.py` (New Module)
**Features:**
- `EnvironmentValidator` class with validation methods
- Validates storage configuration per backend
- Checks required environment variables on package import
- Provides helpful error messages with remediation steps
- Logs safe configuration values for debugging

**Usage:**
```python
from forexgrand_core.env_validator import validate_environment_on_import
validate_environment_on_import()  # Raises EnvironmentError if config missing
```

### 4. Function Documentation

All major functions now have comprehensive docstrings including:
- **Module docstrings**: Explain purpose and list all functions
- **Class docstrings**: Describe usage with type hints
- **Function docstrings**: Include:
  - One-line summary
  - Detailed description
  - Args section with types and descriptions
  - Returns section with types
  - Raises section documenting exceptions
  - Example usage section

**Files Updated:**
- `src/settings.py` - Settings class with all 30+ config variables documented
- `src/data_manager.py` - DataManager class and key methods
- `src/indicators.py` - All 11 indicator functions and 3 factory functions
- `src/main.py` - run_training() function with full examples

**Sample:**
```python
def tf_ma(df: pd.DataFrame, period: int = 12, column: str = "close") -> pd.Series:
    """Simple Moving Average (SMA).
    
    Calculates the average price over a specified period.
    
    Args:
        df: DataFrame containing OHLCV data.
        period: Window size in periods (default: 12). Must be positive.
        column: Column name to calculate MA on (default: 'close'). 
            Options: 'open', 'high', 'low', 'close'.
    
    Returns:
        pd.Series: Moving average values. First (period-1) values are NaN.
    
    Example:
        >>> ma = tf_ma(df, period=20, column='close')
    """
```

### 5. Package Structure & Exports

#### ✅ Updated `__init__.py` Files with Proper Exports

**`src/__init__.py`** (Main package):
- Module docstring with quick start
- Validates environment on import
- Lazy-loads modules to prevent unnecessary dependencies
- Exports: Settings, DataManager, all indicators, training functions
- Uses `__getattr__` for late binding of TensorFlow-dependent modules

**`src/pipeline/__init__.py`**:
- Exports: Trainer, TrainingResult, Evaluator, GenerateTrainData

**`src/models_architecture/__init__.py`**:
- Exports: BaseModel, NoTrainModel

**`src/models_architecture/train_models/__init__.py`**:
- Exports: All model implementations (LSTM, Simple, Conservative, Complex, CNNBiLSTM)

**`src/pipeline/preprocessing/__init__.py`**:
- Exports: PreprocessBase, ExamplePreprocessor

**`src/utils/__init__.py`**:
- Exports: benchmark, FGTesterClient

**`src/storage/clients/__init__.py`**:
- Already had exports (updated documentation)

**`src/storage/__init__.py`**:
- Already had exports (updated documentation)

### 6. Type Hints & Validation

- Full type hints in function signatures
- Environment variable validation on package import
- Helpful error messages guiding users to fix configuration issues
- Lazy module loading to avoid importing unused heavy dependencies

## How to Publish to PyPI

### Step 1: Prepare for Release

```bash
# Update version in pyproject.toml
# Update CHANGELOG or version history in README.md
# Ensure all tests pass
pytest --cov=forexgrand_core tests/
```

### Step 2: Build Distribution Packages

```bash
# Install build tools
pip install build twine

# Create source distribution and wheel
python -m build

# This creates:
# - dist/fg-strategy-modelling-core-0.1.0.tar.gz (source distribution)
# - dist/fg_strategy_modelling_core-0.1.0-py3-none-any.whl (wheel)
```

### Step 3: Test Your Package

```bash
# Install in test environment
pip install dist/fg_strategy_modelling_core-0.1.0-py3-none-any.whl

# Test imports
python -c "from forexgrand_core import Settings, DataManager; print('OK')"
```

### Step 4: Upload to PyPI

#### Option A: Using TestPyPI (Recommended for first-time)

```bash
# Upload to test repository
python -m twine upload --repository testpypi dist/*

# Install from test
pip install -i https://test.pypi.org/simple/ fg-strategy-modelling-core
```

#### Option B: Upload to Production PyPI

```bash
# Create/update ~/.pypirc with your credentials
# Then upload:
python -m twine upload dist/*

# Users can then install with:
pip install fg-strategy-modelling-core
```

### Step 5: Verify Installation

```bash
# In a clean environment
pip install fg-strategy-modelling-core

# Test it works
python -c "from forexgrand_core import Settings; print('Installed successfully')"
```

## Package Information

### Current Configuration

| Item | Value |
|------|-------|
| Package Name | `fg-strategy-modelling-core` |
| Current Version | `0.1.0` |
| Python Requirement | `>=3.9` |
| License | MIT |
| Status | Alpha |

### Dependencies

**Core:**
- boto3>=1.34.0
- pandas>=2.2.0
- pyarrow>=15.0.0
- tensorflow>=2.16.0
- numpy>=1.24.0
- pydantic>=2.0.0

**Optional:**
- dev: pytest, black, mypy, sphinx
- aws: boto3 (included in core)
- gcs: google-cloud-storage
- cloudflare: boto3 (included in core)

### Supported Python Versions

- Python 3.9
- Python 3.10
- Python 3.11
- Python 3.12

## Environment Variables Reference

### Required Variables

```bash
# Data source
DATA_SOURCE=mt5

# Storage configuration
S3_STORAGE_OPTION=minio  # or aws, gcs, cloudflare
S3_ENDPOINT=http://localhost:9000
S3_ACCESS_KEY=your_key
S3_SECRET_KEY=your_secret

# Bucket configuration
S3_BUCKET_NAME=forexgrand
TRAIN_BUCKET_NAME=forexgrand-train
EVAL_BUCKET_NAME=forexgrand-eval
```

See README.md for complete environment variables documentation.

## Key Features of This Package

✅ **Production Ready**
- Full type hints for IDE support
- Comprehensive error handling
- Environment validation on import
- Logging and debugging support

✅ **Well Documented**
- Module docstrings with descriptions
- Function docstrings with examples
- README with quick start and API reference
- Inline code comments for complex logic

✅ **Flexible Configuration**
- All settings via environment variables
- Support for multiple storage backends
- Customizable preprocessing pipelines
- Multiple model architectures

✅ **Proper Packaging**
- pyproject.toml following PEP 517
- setup.py for backward compatibility
- MANIFEST.in for distribution files
- .gitignore configured properly

✅ **Extensible Architecture**
- Abstract base classes for inheritance
- Factory functions for customization
- Lazy module loading for performance
- Clear separation of concerns

## Common Publishing Tasks

### Update Version

1. Edit `pyproject.toml`:
```toml
[project]
version = "0.2.0"
```

2. Update README.md changelog section

3. Create git tag:
```bash
git tag v0.2.0
git push origin v0.2.0
```

### Add New Optional Dependencies

1. Edit `pyproject.toml`:
```toml
[project.optional-dependencies]
new_feature = ["package>=1.0.0"]
```

2. Update install instructions in README.md:
```bash
pip install fg-strategy-modelling-core[new_feature]
```

### Update Metadata

Fields to update in `pyproject.toml`:
- `version` - Increment version number
- `description` - Project description
- `authors` - Maintainer information
- `keywords` - Search keywords
- `classifiers` - PyPI categories

## Next Steps

1. **Add Tests**: Create `tests/` directory with pytest test cases
2. **Setup CI/CD**: Configure GitHub Actions for automated testing and releases
3. **Documentation Site**: Build Sphinx documentation and host on ReadTheDocs
4. **Release Process**: Automate releases with semantic versioning
5. **Community**: Create CONTRIBUTING.md and CODE_OF_CONDUCT.md

## References

- [PyPI Publishing Guide](https://packaging.python.org/tutorials/packaging-projects/)
- [PEP 517](https://www.python.org/dev/peps/pep-0517/) - Build Backend Interface
- [PEP 621](https://www.python.org/dev/peps/pep-0621/) - pyproject.toml Spec
- [setuptools Documentation](https://setuptools.pypa.io/)
- [Twine Documentation](https://twine.readthedocs.io/)

## Support

For issues or questions:
- 📧 Email: dev@forexgrand.com
- 🐛 Issues: [GitHub Issues](https://github.com/forexgrand/fg-strategy-modelling-core/issues)
- 📚 Docs: See README.md and docstrings
