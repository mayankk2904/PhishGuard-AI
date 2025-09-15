FROM python:3.10-slim

# create non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

# copy and install requirements
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# copy app (make sure .git is not copied if you don't want it)
COPY --chown=user . .

# expose port used by spaces
EXPOSE 7860

# run with gunicorn to bind to 0.0.0.0:7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app", "--workers", "1", "--threads", "4", "--timeout", "120"]
