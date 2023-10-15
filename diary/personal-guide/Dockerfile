# official base image
FROM node:18-alpine

#working directory
WORKDIR /app

RUN npm install -g @angular/cli@13

COPY package.json .
COPY package-lock.json .
RUN npm ci
# the other npm part
COPY . .
#start app
CMD ["ng", "serve", "--host", "0.0.0.0"]