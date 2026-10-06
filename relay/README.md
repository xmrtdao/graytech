# AstraGaze brochure — deployment copies

These files are **copies**. The running ones live in the main repo, because the
relay is served from there:

| File | Served from |
|---|---|
| `relay/lib/astragaze-fallback.mjs` | `xmrtdao/../relay/lib/astragaze-fallback.mjs` |
| `relay/public/graytech-offline/index.html` | `xmrtdao/../relay/public/graytech-offline/index.html` |

The reason they are duplicated rather than referenced is that
`xmrtdao/mobilemonero`'s `main` was force-pushed and no longer shares history
with the local branch, so the brochure could not be committed there in a way
anyone could check out and run. Duplication is the lesser evil here, and this
file is the thing that stops it becoming a silent divergence.

## What must also exist, or the page is 401

Two edits live in `relay/server.js`, which is gitignored in the main repo
(`.gitignore` line 34). Both are documented at the top of
`astragaze-fallback.mjs`:

1. **The mount** — `mountAstraGazeFallback(app)` imported from `./lib/`.
2. **The auth allowlist** — `req.path === '/graytech' || req.path === '/astragaze'`
   in the middleware that guards every tunnel request.

The second is the one that matters. That middleware treats any request carrying
`cf-ray` or `cf-connecting-ip` as external and demands credentials, so without the
allowlist entry the brochure returns 401 to the public — and the one page whose
job is to be readable during an outage would be unreadable during an outage.

## Verify after any change, from outside this machine

```
curl -sI https://astragaze.mobilemonero.com/graytech | head -1
```

Expect `HTTP/2 200`. Then confirm the header:

```
curl -sI https://astragaze.mobilemonero.com/graytech | grep -i astragaze
```

Expect `x-astragaze-fallback: true`. If the header is absent, you are looking at
the face service or an error page, not the brochure.

## How it is wired

```
astragaze.mobilemonero.com   CNAME, proxied
  -> 61492f26-c8f8-45d2-be65-ffb7340683fa.cfargotunnel.com
  -> cloudflared tunnel (ingress rule added to ~/.cloudflared/config.yml)
  -> localhost:8080           relay
  -> /graytech or /astragaze  this page
```

It is deliberately **not** `graytech.mobilemonero.com`, which routes to port 8090
and is the service that goes down. And not `relay.mobilemonero.com`, which is
behind Cloudflare Access and 401s anonymous visitors.

## If you change the design

The colour tokens and type stack are lifted from `graytech/graytech/server.py`'s
`:root` block so the brochure and the console are visibly the same brand. Change
them in both places or they drift.