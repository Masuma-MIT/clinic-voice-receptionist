# Browser and telephone runtime coordination review

This update follows the observed browser failure to stop assistant speech when the caller interrupted. It preserves the installed reference/quantity correction and contextual Qwen interpretation layer.

## Changes

- Browser audio generation and audible playback are tracked separately. The browser acknowledges completion only after the stream ends and its scheduled audio has drained. A gap between audio chunks is not treated as finished speech.
- Silero speech-start events stop current and queued browser speech without waiting for a final transcript. Caller transcript fragments remain available to the normal turn assembler.
- A response becoming ready while the caller is already speaking cannot start over that caller. Appointment state is retained for the subsequent caller turn.
- Browser playback generations reject late audio and stop/end events from an older response. Closed-connection callbacks cannot reset a replacement browser conversation.
- Telephone speech-start events interrupt sending and clear the Twilio audio buffer. Cancelling the child audio task preserves the caller-processing task and its queued input.
- Twilio marks distinguish completed sending from completed playback. Marks returned after clearing old audio cannot finish a newer response.
- Both interfaces bind short confirmations to a relevant readback delivered before the caller started that reply. An early yes repeats the readback without authorising a write.
- A superseded telephone response retains its prepared proposal. The readback gate prevents that unheard proposal from being committed by an old yes. The previous offline expectation that discarded the proposal has been updated to this policy.
- An unverified browser processing failure no longer asks the caller to blindly repeat a possibly completed transaction.

## Evidence and boundaries

The installer executes 83 new runtime/controller assertions with extracted runtime functions, simulated transports and an in-memory database. The packaging review separately executed 33 assertions against the actual browser JavaScript using a simulated DOM, AudioContext and WebSocket. Previous controller, enquiry, booking-sequence, lookup, natural-response, semantic-protocol and reference/quantity checks were also run. These suites overlap and must not be presented as independent live conversations or as a measured accuracy percentage.

There were no live API calls, account changes, model downloads or real database writes during packaging or installation. Browser/telephone recordings, real speaker-model accuracy, network behaviour, provider latency and physical interruption behaviour still require live checks. This update does not establish that all remaining speaker false positives or response delays are resolved. It does not change speaker thresholds or claim voice-based patient authentication. Speech detection and microphone echo cancellation can still require acoustic evaluation.

## Before freezing the revision

1. Restart the browser and phone processes after installation; refresh the browser page to load its updated JavaScript. Keep the configured ngrok endpoint and Twilio webhook.
2. Verify that an existing-booking request without an ID asks for a reference. Supply a reference naturally and check the returned record.
3. Interrupt a longer browser response near its beginning and later during playback; verify that old speech stops and the new request is retained. Repeat on the telephone.
4. Complete booking, lookup, time/date rescheduling and cancellation using explicitly designated test appointments; read back each result in the active Supabase project.
5. Test an early yes, a later yes after the current readback, a changed preference, repeated yes, and a new conversation. Confirm that unintended or duplicate writes do not occur.
6. Review fresh latency and conversation logs from both interfaces. Record any remaining speaker clarification errors with their audio evidence before changing speaker policy.
7. Commit and push the verified source, updated hashes and validation notes. Do not publish credentials, recordings or private call logs.

The technical basis for telephone completion marks and clearing buffered audio is Twilio's official Media Streams WebSocket documentation: https://www.twilio.com/docs/voice/media-streams/websocket-messages
