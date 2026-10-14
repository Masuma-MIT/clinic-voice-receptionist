-- Run ONLY in the demonstration database after 01 and 02.
-- Exercise database tools as service_role, then roll back record changes.
-- Sequence numbers can advance despite rollback; no test appointment is retained.
BEGIN;
SET LOCAL ROLE service_role;
SET LOCAL search_path = pg_catalog, public;
DO $verify$
DECLARE
 first_slot bigint;
 next_slot bigint;
 doctor bigint;
 service bigint;
 appointment bigint;
 result jsonb;
 before_history bigint;
BEGIN
 IF (SELECT count(*) FROM public.services) <> 13
    OR (SELECT count(*) FROM public.doctors) <> 65
    OR (SELECT count(*) FROM public.doctor_services) <> 67
    OR (SELECT count(*) FROM public.clinic_knowledge) <> 19 THEN
   RAISE EXCEPTION 'Reference inventory mismatch';
 END IF;
 IF EXISTS (SELECT 1 FROM public.slots WHERE appointment_date <=
     (CURRENT_TIMESTAMP AT TIME ZONE 'Australia/Sydney')::date) THEN
   RAISE EXCEPTION 'Demo slots are no longer all in the future; inspect before testing';
 END IF;
 SELECT s.slot_id,s.doctor_id,s.service_id INTO first_slot,doctor,service
 FROM public.slots s JOIN public.services sv ON sv.service_id=s.service_id
 WHERE sv.service_name='Medical Certificate Consultation'
   AND s.consultation_mode='in_person' AND s.available
 ORDER BY s.appointment_date,s.appointment_time,s.slot_id LIMIT 1;
 SELECT s.slot_id INTO next_slot FROM public.slots s
 WHERE s.doctor_id=doctor AND s.service_id=service AND s.slot_id<>first_slot
   AND s.consultation_mode='in_person' AND s.available
 ORDER BY s.appointment_date,s.appointment_time,s.slot_id LIMIT 1;
 IF first_slot IS NULL OR next_slot IS NULL THEN RAISE EXCEPTION 'Required demo slots missing'; END IF;
 SELECT count(*) INTO before_history FROM public.appointment_change_history;
 result := public.book_appointment_with_reason_atomic('Demonstration Test Patient',first_slot,'Setup verification only');
 IF result->>'success' IS DISTINCT FROM 'true' THEN RAISE EXCEPTION 'Booking failed: %',result; END IF;
 appointment := (result->>'appointment_id')::bigint;
 IF (SELECT available FROM public.slots WHERE slot_id=first_slot) THEN
   RAISE EXCEPTION 'Booking failed to reserve the slot';
 END IF;
 result := public.book_appointment_atomic('Demonstration Duplicate Patient',first_slot);
 IF result->>'success' IS DISTINCT FROM 'false' THEN RAISE EXCEPTION 'Occupied-slot protection failed'; END IF;
 UPDATE public.appointments SET patient_name='Demonstration Corrected Patient' WHERE appointment_id=appointment;
 IF NOT EXISTS (SELECT 1 FROM public.appointments WHERE appointment_id=appointment
                AND patient_name='Demonstration Corrected Patient' AND updated_at IS NOT NULL) THEN
   RAISE EXCEPTION 'Name update or stamping failed';
 END IF;
 INSERT INTO public.medical_certificate_requests
   (appointment_id,consultation_mode,insurance_status,prototype_fee_aud,
    certificate_purpose,requested_date_range,prototype_only)
 VALUES (appointment,'in_person','insured',0,'Demonstration only','Demonstration only',true)
 ON CONFLICT (appointment_id) DO UPDATE SET certificate_purpose=EXCLUDED.certificate_purpose;
 IF NOT EXISTS (SELECT 1 FROM public.medical_certificate_requests WHERE appointment_id=appointment) THEN
   RAISE EXCEPTION 'Certificate intake persistence failed';
 END IF;
 result := public.reschedule_appointment_atomic(appointment,next_slot);
 IF result->>'success' IS DISTINCT FROM 'true' THEN RAISE EXCEPTION 'Reschedule failed: %',result; END IF;
 IF NOT (SELECT available FROM public.slots WHERE slot_id=first_slot)
    OR (SELECT available FROM public.slots WHERE slot_id=next_slot) THEN
   RAISE EXCEPTION 'Reschedule slot states incorrect';
 END IF;
 IF NOT EXISTS (SELECT 1 FROM public.appointment_overview WHERE appointment_id=appointment
                AND display_status='rescheduled' AND reschedule_count=1) THEN
   RAISE EXCEPTION 'Overview, manifest access or reschedule stamping failed';
 END IF;
 result := public.cancel_appointment_atomic(appointment);
 IF result->>'success' IS DISTINCT FROM 'true' THEN RAISE EXCEPTION 'Cancel failed: %',result; END IF;
 IF NOT (SELECT available FROM public.slots WHERE slot_id=next_slot) THEN
   RAISE EXCEPTION 'Cancellation failed to release the slot';
 END IF;
 IF (SELECT count(*) FROM public.appointment_change_history) <> before_history+4 THEN
   RAISE EXCEPTION 'Expected booking, name-update, reschedule and cancellation history';
 END IF;
 IF has_table_privilege('anon','public.appointments','SELECT')
    OR has_table_privilege('authenticated','public.appointments','UPDATE')
    OR has_function_privilege('anon','public.book_appointment_atomic(text,bigint)','EXECUTE') THEN
   RAISE EXCEPTION 'Unexpected direct-client access';
 END IF;
 RAISE NOTICE 'DEMONSTRATION DATABASE CHECKS PASSED: reference inventory, booking, occupied slot, name update, certificate intake, reschedule, cancellation, history, overview and access checks';
END;
$verify$;
ROLLBACK;
