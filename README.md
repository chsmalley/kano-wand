# kano-wand

## Spell and treat event log

When `app.py` runs, it appends every received spell and each treat dispensed
to `event_log.csv` in the project directory. The CSV includes a UTC ISO 8601
timestamp, event type, spell, confidence, wand ID, and treat source. The file
is opened in append mode, so existing events are retained across restarts.

Set `EVENT_LOG_PATH` to write the log somewhere else. For example:

```sh
EVENT_LOG_PATH=/path/to/events.csv python app.py
```

The web page includes a live chart of spell-triggered tricks and dispensed
treats over the rolling past 24 hours, grouped into 15-minute UTC intervals.
The chart refreshes every 15 seconds; trick counts are derived from the spell
to trick mapping in `app.py`.