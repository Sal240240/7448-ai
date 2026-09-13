# Frontend image: build the static bundle, serve it from nginx.
#
# VITE_API_BASE is baked in at build time (Vite inlines env vars into the
# bundle), so the API origin must be supplied as a build arg rather than a
# runtime environment variable:
#     docker build --build-arg VITE_API_BASE=https://api.example.com ...
# Leave it empty to have the browser call /api on the same origin, which is
# what the nginx proxy block below expects.
FROM node:22-alpine AS build

WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci || npm install

COPY . .
ARG VITE_API_BASE=""
ENV VITE_API_BASE=$VITE_API_BASE
RUN npm run build

FROM nginx:1.27-alpine AS runtime
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
