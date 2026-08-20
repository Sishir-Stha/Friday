# Friday database

## Long-term memories

The existing PostgreSQL `memories` table stores durable memory records with a
type, content, importance from 1 through 10, optional source message, active
status, and creation/update timestamps. The existing
`idx_memories_active_importance` index supports active-memory recall.

`MemoryService.remember()` creates records, while `recall()` returns active
records with an optional type filter. `forget()` sets `is_active` to false and
`restore()` sets it back to true; neither operation physically deletes a row.
`get()` can retrieve active or inactive records for management and debugging.
