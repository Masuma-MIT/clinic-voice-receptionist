# Database configuration and optional independent setup

## Current maintained system

The updated browser and Twilio applications now share the original research Supabase project, with its existing project name retained. Four newer appointments, their history and two certificate requests were consolidated into that project while preserving all 20 original appointments. Supplied verification confirmed 24 appointments, 22,366 slots and 10 history records; both appointment triggers were enabled and the checked application permissions were present. The local connection switch completed with a private configuration backup. Telephone gateway startup and public preflight passed after the switch; a complete live appointment operation against the consolidated database is still pending.

The former separate demonstration project is no longer configured as the application database. Its deletion has not been confirmed. Private migration archives remain inside the original project and are not public repository contents. See the consolidation entry in `DATABASE_SETUP_VALIDATION.txt` for the conflict policy and evidence boundaries.

## Optional independent reviewer setup

The scripts below prepare a separate empty database only for independent installation. They are not needed to operate the existing original project and must not be run against it. Use a new, empty Supabase project and run them as the project database administrator.

## Installation order

1. Run `database/01_schema.sql` in the new project's SQL Editor.
2. Run `database/02_demo_reference_data.sql`.
3. Run `database/03_verify_demo_database.sql`. Review the SQL Editor messages for the success notice; any exception means verification failed. If an error leaves a transaction open, run `ROLLBACK;` before retrying after investigation.
4. Put the new project's Supabase URL and server-side secret/service-role key into the reviewing computer's local `.env`, using the repository's `.env.example`. Never commit this completed file or distribute a database password in the public repository.
5. Continue with the browser or telephone setup described in the repository README. Both interfaces use the shared receptionist controller and database.

The schema and seed scripts run in transactions and refuse existing application objects or populated seed tables. They are initial installation scripts, not live migrations or rerunnable reset scripts. Do not delete existing data to bypass their guards.

## Included database objects

The package recreates ten public tables: services, doctors, doctor_services, slots, appointments, clinic_knowledge, appointment_change_history, medical_certificate_requests, synthetic_test_runs and synthetic_test_scenario_results. The last two are empty evidence-storage tables; historical test records are not imported.

It includes the 26 supplied constraints, the supplied non-constraint indexes, two appointment triggers, six application/trigger function definitions, and the appointment_overview view. Identity columns generate their own seven sequences. Exact bigint identity bounds are taken from the string-valued column metadata, avoiding rounded JSON sequence bounds.

The overview's external manifest dependency is recreated in clinic_backup_details_v1 with the two exported column definitions. Its original capture timestamp is retained as provenance. The migration text is left null because it was not supplied. No backup appointment data are included.

Supabase's exported rls_auto_enable event-trigger function is intentionally not recreated: it belongs to platform schema management and is not called by the receptionist. Every application table explicitly enables row-level security.

## Reference data and demonstration slots

The supplied reference export contains 13 services, 65 doctors, 67 doctor-service mappings and 19 clinic-knowledge records. Mapping references were checked for consistency. The knowledge includes cafeteria, playground, prayer area and the other supplied administrative information. Existing appointments, appointment history, certificate-intake records, participant transcripts, audio, credentials and test-run payloads are excluded.

New demonstration slots are generated for the next 30 dates in Australia/Sydney. Each doctor-service mapping receives four in-person times daily. Medical Certificate Consultation additionally receives audio and video times. Those times are distinct because the source slot uniqueness constraint excludes consultation mode. With the supplied mappings, the seed creates 8,400 new slots. These slots are invented demonstration availability, not exported clinic schedules or additional research results. The original 14,432-slot inventory is not reproduced.

## Deliberate portability and permission differences

The metadata export supplied base PostgreSQL types but did not supply type precision, scale or length modifiers. The setup uses those base types, retaining defaults, nullability, identity settings and constraints. It is therefore a functional demonstration reconstruction, not a byte-identical pg_dump or a complete restoration of the live database.

The new project grants server-side service_role the operations needed by the submitted controller. No direct anonymous or authenticated-client table access or appointment-function execution is granted. No RLS policies are added; service_role must retain Supabase's BYPASSRLS behaviour. Keep its key on the server.

The live export showed no SELECT/INSERT/UPDATE access for service_role on medical_certificate_requests, including its column permissions. The controller attempts an upsert into that table after successful booking and catches failure as an intake warning. This demonstration setup grants SELECT, INSERT and UPDATE plus identity-sequence access so intake metadata can be saved. This is an explicitly documented demonstration permission correction; the live database and Python source were not modified.

The working database's broad TRUNCATE/REFERENCES/TRIGGER/MAINTAIN grants are not copied. Trigger functions are restricted to the owner; the history function retains its supplied SECURITY DEFINER implementation. The four appointment RPC functions are executable by service_role only. Evidence-storage tables remain empty without service-role write grants because the running receptionist does not use them.

## Validation status

The checks below describe successive packaging and demonstration stages. The user reported successful schema installation, demonstration-data seeding, and corrected rollback verification on Supabase. Subsequent output confirmed the demonstration reference counts and database connectivity without writes. The user also reported successful browser interaction against this demonstration database. Telephone gateway startup, public preflight readiness, and speaker-worker readiness were confirmed in supplied output. A telephone conversation exposed incomplete enquiry coverage, which was subsequently addressed by a repair reporting 42 passing offline enquiry cases; no further telephone call was performed after that repair. These observations do not establish complete telephone validation of the revised application or successful execution on another computer. See `DATABASE_SETUP_VALIDATION.txt` for the recorded scope.

The SQL scripts and six supplied PL/pgSQL function bodies were checked with a PostgreSQL parser. Reference-data counts and foreign-key mappings were checked offline. These checks do not prove installation, grants, triggers or transactions execute correctly on Supabase.

The rollback-based verification script is supplied to test those behaviours in the new project. It checks booking, refusal to book an occupied slot, name update, certificate-intake persistence, rescheduling, cancellation, history, overview access and selected access restrictions. Record changes are rolled back; PostgreSQL sequence numbers can advance despite rollback. It does not call an LLM, speech API, browser or Twilio and does not establish complete application performance.

A successful database check must be followed by environment installation, model setup and browser/telephone startup checks before independent execution can be confirmed. Authorised Deepgram and Qwen access is still required. Telephone operation additionally needs Twilio and a reachable HTTPS/WebSocket endpoint. Telephone multi-model verification needs its separate environment, model assets and authorised Hugging Face access. Those are not supplied by this database package.
