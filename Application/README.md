# Start the application

For downloading the requirements use this command

```
pip3 install requirements.txt
```

Use this command to start the python application in a terminal

```
python3 app.py
```

Then start the redis backend in seperate terminal

```
redis-server
```

Then start the celery worker

```
./worker.sh
```