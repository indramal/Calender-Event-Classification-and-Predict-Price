FROM public.ecr.aws/lambda/python:3.8

# Install the function's dependencies using file requirements.txt
# from your project folder.

COPY requirements.txt  .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

COPY Regression_Models ${LAMBDA_TASK_ROOT}/Regression_Models

COPY Model ${LAMBDA_TASK_ROOT}/Model

COPY bill-extractor-586a1-firebase-adminsdk-bnhw2-c376f19014.json ${LAMBDA_TASK_ROOT}

# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ]