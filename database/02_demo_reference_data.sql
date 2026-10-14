-- Seed ONLY the new demonstration project after 01_schema.sql.

-- Reference records come from supplied export. Future slots below are newly generated demo data.

BEGIN;

SET LOCAL search_path = pg_catalog, public;

DO $guard$
BEGIN
 IF EXISTS (SELECT 1 FROM public.services) OR EXISTS (SELECT 1 FROM public.doctors)
    OR EXISTS (SELECT 1 FROM public.doctor_services) OR EXISTS (SELECT 1 FROM public.clinic_knowledge)
    OR EXISTS (SELECT 1 FROM public.slots) OR EXISTS (SELECT 1 FROM public.appointments) THEN
   RAISE EXCEPTION 'STOP: seed requires empty application tables.';
 END IF;
END;
$guard$;

INSERT INTO public."services" ("active", "department", "service_id", "description", "service_name") OVERRIDING SYSTEM VALUE VALUES
(true, 'Primary Care', 1, 'General health concerns, common illnesses, initial assessment, routine consultation and referral when required.', 'General Practice'),
(true, 'Primary Care', 2, 'Short general practice consultation for assessment related to a medical certificate.', 'Medical Certificate Consultation'),
(true, 'Specialist Medicine', 3, 'Heart-related specialist consultation, including cardiovascular concerns and follow-up.', 'Cardiology'),
(true, 'Mental Health', 4, 'Mental health assessment and psychiatric consultation.', 'Psychiatry'),
(true, 'Allied Health', 5, 'Dietary and nutrition support for general health, weight management, diabetes and other nutrition-related concerns.', 'Dietetics and Nutrition'),
(true, 'Allied Health', 6, 'Exercise and rehabilitation support for injury, chronic conditions, mobility and safe physical activity.', 'Exercise Physiology'),
(true, 'Specialist Medicine', 7, 'Skin, hair and nail-related specialist consultation.', 'Dermatology'),
(true, 'Specialist Medicine', 8, 'Health assessment and specialist care for children.', 'Paediatrics'),
(true, 'Specialist Medicine', 9, 'Ear, nose and throat-related specialist consultation.', 'ENT'),
(true, 'Specialist Medicine', 10, 'Specialist consultation for neurological symptoms and conditions.', 'Neurology'),
(true, 'Specialist Medicine', 11, 'Specialist consultation for bones, joints, muscles and musculoskeletal problems.', 'Orthopaedics'),
(true, 'Specialist Medicine', 12, 'Specialist consultation for women''s reproductive and gynaecological health.', 'Gynaecology'),
(true, 'Specialist Medicine', 13, 'Specialist consultation for hormonal, metabolic and endocrine-related conditions.', 'Endocrinology');

INSERT INTO public."doctors" ("doctor_id", "specialty", "service_id", "doctor_name", "consultant_gender") OVERRIDING SYSTEM VALUE VALUES
(1, 'Cardiology', 3, 'Dr Sarah Lee', NULL),
(3, 'General Practice', 1, 'Dr Emily Brown', NULL),
(4, 'General Practice', 1, 'Dr Michael Chen', NULL),
(5, 'Psychiatry', 4, 'Dr Olivia Wilson', NULL),
(6, 'Dermatology', 7, 'Dr Daniel Kim', NULL),
(7, 'Paediatrics', 8, 'Dr Sophia Patel', NULL),
(8, 'ENT', 9, 'Dr James Taylor', NULL),
(9, 'Neurology', 10, 'Dr Hannah Martin', NULL),
(10, 'Orthopaedics', 11, 'Dr Robert Evans', NULL),
(11, 'Gynaecology', 12, 'Dr Amelia Clark', NULL),
(12, 'Endocrinology', 13, 'Dr Ethan Walker', NULL),
(13, 'Dietetics and Nutrition', 5, 'Ms Chloe Harris', NULL),
(14, 'Exercise Physiology', 6, 'Mr Liam Scott', NULL),
(15, 'General Practice and preventive health', NULL, 'Dr Maya Thompson', 'female'),
(16, 'General Practice and chronic-care coordination', NULL, 'Dr Daniel Okafor', 'male'),
(17, 'Cardiology with an interest in heart rhythm care', NULL, 'Dr Aisha Rahman', 'female'),
(18, 'Cardiology with an interest in heart failure care', NULL, 'Dr James Walker', 'male'),
(19, 'Dermatology with an interest in inflammatory skin conditions', NULL, 'Dr Sofia Martinez', 'female'),
(20, 'Dermatology with an interest in skin-lesion assessment', NULL, 'Dr Ethan Williams', 'male'),
(21, 'Clinical nutrition and metabolic health', NULL, 'Dr Priya Nair', 'female'),
(22, 'Dietetics and gastrointestinal nutrition', NULL, 'Dr Lucas Bennett', 'male'),
(23, 'Ear nose and throat care with an interest in sinus conditions', NULL, 'Dr Hannah Lee', 'female'),
(24, 'Ear nose and throat care with an interest in hearing and balance', NULL, 'Dr Omar Khalid', 'male'),
(25, 'Endocrinology with an interest in diabetes care', NULL, 'Dr Grace Kim', 'female'),
(26, 'Endocrinology with an interest in thyroid care', NULL, 'Dr Noah Patel', 'male'),
(27, 'Exercise physiology and rehabilitation', NULL, 'Dr Chloe Adams', 'female'),
(28, 'Exercise physiology and chronic-condition support', NULL, 'Dr Samuel Wright', 'male'),
(29, 'Gynaecology and reproductive health', NULL, 'Dr Layla Hassan', 'female'),
(30, 'Gynaecology with an interest in minimally invasive care', NULL, 'Dr Thomas Nguyen', 'male'),
(31, 'General Practice and medical-certificate assessment', NULL, 'Dr Emma Clarke', 'female'),
(32, 'General Practice and medical-certificate assessment', NULL, 'Dr Arjun Mehta', 'male'),
(33, 'Neurology with an interest in headache and migraine care', NULL, 'Dr Amelia Scott', 'female'),
(34, 'Neurology with an interest in movement disorders', NULL, 'Dr Benjamin Silva', 'male'),
(35, 'Orthopaedics with an interest in upper-limb conditions', NULL, 'Dr Zoe Campbell', 'female'),
(36, 'Orthopaedics with an interest in hip and knee conditions', NULL, 'Dr Marcus Evans', 'male'),
(37, 'Paediatrics and child development', NULL, 'Dr Isabella Moore', 'female'),
(38, 'Paediatrics with an interest in adolescent health', NULL, 'Dr Yusuf Ali', 'male'),
(39, 'Psychiatry with an interest in anxiety and mood disorders', NULL, 'Dr Natalie Brooks', 'female'),
(40, 'Psychiatry with an interest in adult mental health', NULL, 'Dr Henry Chen', 'male'),
(41, 'General Practice and women''s health', NULL, 'Dr Olivia Morgan', 'female'),
(42, 'General Practice and men''s health', NULL, 'Dr Liam Anderson', 'male'),
(43, 'Cardiology with an interest in preventive cardiac care', NULL, 'Dr Fatima Noor', 'female'),
(44, 'Cardiology with an interest in coronary care', NULL, 'Dr Alexander Reed', 'male'),
(45, 'Dermatology with an interest in paediatric skin conditions', NULL, 'Dr Emily Zhao', 'female'),
(46, 'Dermatology with an interest in hair and scalp conditions', NULL, 'Dr Hamza Siddiqui', 'male'),
(47, 'Clinical nutrition and maternal nutrition', NULL, 'Dr Mia Roberts', 'female'),
(48, 'Sports nutrition and weight-management support', NULL, 'Dr Isaac Mensah', 'male'),
(49, 'Ear nose and throat care with an interest in voice disorders', NULL, 'Dr Sarah Ibrahim', 'female'),
(50, 'Ear nose and throat care with an interest in paediatric ENT', NULL, 'Dr William Park', 'male'),
(51, 'Endocrinology with an interest in reproductive hormones', NULL, 'Dr Ruby Fernando', 'female'),
(52, 'Endocrinology with an interest in bone metabolism', NULL, 'Dr George Wilson', 'male'),
(53, 'Exercise physiology and cardiac rehabilitation', NULL, 'Dr Ava Collins', 'female'),
(54, 'Exercise physiology and musculoskeletal rehabilitation', NULL, 'Dr Jack Murphy', 'male'),
(55, 'Gynaecology with an interest in adolescent health', NULL, 'Dr Noor Begum', 'female'),
(56, 'Gynaecology with an interest in reproductive medicine', NULL, 'Dr Christopher Hall', 'male'),
(57, 'General Practice and medical-certificate assessment', NULL, 'Dr Sophie Taylor', 'female'),
(58, 'General Practice and medical-certificate assessment', NULL, 'Dr Mohammed Rahman', 'male'),
(59, 'Neurology with an interest in epilepsy care', NULL, 'Dr Charlotte Green', 'female'),
(60, 'Neurology with an interest in neuromuscular conditions', NULL, 'Dr Ryan Cooper', 'male'),
(61, 'Orthopaedics with an interest in foot and ankle conditions', NULL, 'Dr Lily Johnson', 'female'),
(62, 'Orthopaedics with an interest in spine conditions', NULL, 'Dr Adam Brown', 'male'),
(63, 'Paediatrics with an interest in respiratory health', NULL, 'Dr Sophia Evans', 'female'),
(64, 'Paediatrics with an interest in childhood allergy', NULL, 'Dr David Khan', 'male'),
(65, 'Psychiatry with an interest in perinatal mental health', NULL, 'Dr Eleanor White', 'female'),
(66, 'Psychiatry with an interest in older-adult mental health', NULL, 'Dr Michael Grant', 'male');

INSERT INTO public."doctor_services" ("doctor_id", "service_id") VALUES
(1, 3),
(3, 1),
(3, 2),
(4, 1),
(4, 2),
(5, 4),
(6, 7),
(7, 8),
(8, 9),
(9, 10),
(10, 11),
(11, 12),
(12, 13),
(13, 5),
(14, 6),
(15, 1),
(16, 1),
(17, 3),
(18, 3),
(19, 7),
(20, 7),
(21, 5),
(22, 5),
(23, 9),
(24, 9),
(25, 13),
(26, 13),
(27, 6),
(28, 6),
(29, 12),
(30, 12),
(31, 2),
(32, 2),
(33, 10),
(34, 10),
(35, 11),
(36, 11),
(37, 8),
(38, 8),
(39, 4),
(40, 4),
(41, 1),
(42, 1),
(43, 3),
(44, 3),
(45, 7),
(46, 7),
(47, 5),
(48, 5),
(49, 9),
(50, 9),
(51, 13),
(52, 13),
(53, 6),
(54, 6),
(55, 12),
(56, 12),
(57, 2),
(58, 2),
(59, 10),
(60, 10),
(61, 11),
(62, 11),
(63, 8),
(64, 8),
(65, 4),
(66, 4);

INSERT INTO public."clinic_knowledge" ("title", "active", "content", "category", "created_at", "knowledge_id") OVERRIDING SYSTEM VALUE VALUES
('Clinic Opening Hours', true, 'Open 24 hours a day, 7 days a week.', 'Hours', '2026-08-18T08:07:16.164036+00:00', 1),
('Clinic Contact Information', true, 'Patients can contact the clinic reception by phone during opening hours. Appointment enquiries, cancellations and rescheduling requests can be handled through reception.', 'Contact', '2026-08-18T08:07:16.164036+00:00', 2),
('Clinic Location', true, 'The clinic is located within the local metropolitan area. Patients should confirm the exact address before attending their appointment.', 'Location', '2026-08-18T08:07:16.164036+00:00', 3),
('Appointment Booking Policy', true, 'Appointments are subject to current provider and slot availability. Patients should provide their name and preferred appointment date and time when booking.', 'Appointments', '2026-08-18T08:07:16.164036+00:00', 4),
('Cancellation Policy', true, 'Patients may request cancellation of a booked appointment. Once cancelled, the appointment slot becomes available for another patient.', 'Appointments', '2026-08-18T08:07:16.164036+00:00', 5),
('Rescheduling Policy', true, 'Patients may request to move an existing appointment to another available slot. The new appointment time must be available before the current booking is changed.', 'Appointments', '2026-08-18T08:07:16.164036+00:00', 6),
('What to Bring', true, 'Patients should bring appropriate identification and any documents specifically requested by the clinic or provider for their appointment.', 'Documents', '2026-08-18T08:07:16.164036+00:00', 7),
('Clinic Services', true, 'The clinic provides General Practice, Medical Certificate Consultation, Cardiology, Psychiatry, Dietetics and Nutrition, Exercise Physiology, Dermatology, Paediatrics, ENT, Neurology, Orthopaedics, Gynaecology and Endocrinology services.', 'Services', '2026-08-18T08:07:16.164036+00:00', 8),
('Patient Waiting Area', true, 'The clinic has a patient waiting area for patients and accompanying visitors.', 'Facilities', '2026-08-27T06:47:50.97245+00:00', 9),
('Private Breastfeeding Area', true, 'The clinic has a private breastfeeding and infant-care area.', 'Facilities', '2026-08-27T06:47:50.97245+00:00', 10),
('Onsite Pharmacy', true, 'The clinic has an onsite pharmacy available 24 hours a day, 7 days a week.', 'Facilities', '2026-08-27T06:47:50.97245+00:00', 11),
('Public Cafeteria', true, 'The clinic has a public cafeteria available daily from 6:00 AM to 10:00 PM.', 'Facilities', '2026-08-27T06:47:50.97245+00:00', 12),
('Children''s Playground', true, 'The clinic has a children''s playground and play area.', 'Facilities', '2026-08-27T06:47:50.97245+00:00', 13),
('Emergency Department', true, 'The prototype clinic emergency department is available 24 hours a day, 7 days a week.', 'Emergency', '2026-09-21T05:56:23.992536+00:00', 14),
('Prototype Ambulance Service', true, 'The prototype clinic lists an internal demonstration ambulance extension of 9000. It does not place a real ambulance call. For an immediate or life-threatening emergency in Australia, call 000.', 'Emergency', '2026-09-21T05:56:23.992536+00:00', 15),
('Prayer Area', true, 'The clinic has a quiet multi-faith prayer area available 24 hours a day.', 'Facilities', '2026-09-21T05:56:23.992536+00:00', 16),
('Children''s Play Area', true, 'The clinic has a children''s play area available daily from 8:00 AM to 8:00 PM. Children must remain supervised by a responsible adult.', 'Facilities', '2026-09-21T05:56:23.992536+00:00', 17),
('Medical Certificate Consultation Options', true, 'Medical certificate assessment appointments may be in person or online. Online appointments may use video or audio. A booking is for a clinician assessment. The clinician decides whether a medical certificate can be issued.', 'Medical Certificates', '2026-09-21T05:56:23.992536+00:00', 18),
('Medical Certificate Prototype Fee', true, 'For this research prototype, the displayed medical-certificate assessment fee is AUD 0 with medical insurance and AUD 45 without medical insurance. These are fictional prototype values and no payment is processed.', 'Medical Certificates', '2026-09-21T05:56:23.992536+00:00', 19);

SELECT pg_catalog.setval(pg_catalog.pg_get_serial_sequence('public.services', 'service_id'), (SELECT max("service_id") FROM public."services"), true);

SELECT pg_catalog.setval(pg_catalog.pg_get_serial_sequence('public.doctors', 'doctor_id'), (SELECT max("doctor_id") FROM public."doctors"), true);

SELECT pg_catalog.setval(pg_catalog.pg_get_serial_sequence('public.clinic_knowledge', 'knowledge_id'), (SELECT max("knowledge_id") FROM public."clinic_knowledge"), true);

-- 30 future Sydney dates; each doctor-service mapping has four in-person times.
-- Medical-certificate service also has audio and video times, kept distinct because
-- the source unique constraint does not include consultation_mode.
INSERT INTO public.slots(doctor_id, service_id, appointment_date, appointment_time, available, consultation_mode)
SELECT ds.doctor_id, ds.service_id,
       (CURRENT_TIMESTAMP AT TIME ZONE 'Australia/Sydney')::date + days.n,
       times.appointment_time, true, times.consultation_mode
FROM public.doctor_services ds
CROSS JOIN generate_series(1,30) AS days(n)
CROSS JOIN (VALUES
  ('09:00'::time, 'in_person'::text), ('10:00'::time,'in_person'),
  ('14:00'::time, 'in_person'), ('15:00'::time,'in_person'),
  ('11:00'::time, 'audio'), ('13:00'::time,'video')
) AS times(appointment_time, consultation_mode)
JOIN public.services sv ON sv.service_id=ds.service_id
WHERE times.consultation_mode='in_person'
   OR sv.service_name='Medical Certificate Consultation';

COMMIT;
