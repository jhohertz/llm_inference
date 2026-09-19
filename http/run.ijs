NB. ============================================================
NB. run.ijs - HTTP server ENTRY POINT.
NB. http/server.ijs is a DORMANT library (verbs only, no auto-launch);
NB. this file loads it and explicitly launches via server_run ''.
NB.
NB. usage:  jconsole http/run.ijs [MODEL]   (shell helper scripts/llm_server.sh)
NB.         jconsole http/server.ijs MODEL  no longer launches (library only).
NB. ============================================================
require 'llm/inference/http/server'
cocurrent <'inference'
server_run ''
