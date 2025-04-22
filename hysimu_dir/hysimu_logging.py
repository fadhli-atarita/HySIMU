# ======================================================================= #
"""
hysimu_logging
------
A function to centralize logging for hysimu modules. Except for joblib
verbose
"""
# ======================================================================= #
# IMPORT PACKAGES
# ======================================================================= #
import logging


# ======================================================================= #
# LOGGING FUNCTION
# ======================================================================= #
def main(
    work_dir,
    project_name
):
    """
    Log hysimu outputs

    Parameters:
        - work_dir (str): working directory
        - project_name (str): project name as identifier

    Returns:
        - logfile
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                f"{work_dir}/{project_name}_logfile.log", mode='w'
            ),
            logging.StreamHandler()
        ]  # set log handler and name
    )

    logger = logging.getLogger()

    return logger


# ======================================================================= #
# INITIALIZE MAIN FUNCTION
# ======================================================================= #
if __name__ == "__main__":
    main()


# ======================================================================= #
# ======================================================================= #
