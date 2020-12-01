FROM python:3


WORKDIR /usr/src/aavail_predict

# install python packages
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy all dir and files to image 
# ***TBD copy only the necessaries directories and files
# not very optimized yet 
COPY . .


# start the server
CMD [ "./start-server.sh"  ]

# the server is exposed on port 5000
EXPOSE 5000/tcp