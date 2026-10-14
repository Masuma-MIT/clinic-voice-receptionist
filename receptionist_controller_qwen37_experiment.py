# CALL_FLOW_STABILITY_V4_20261012
# SPOKEN_DATE_VERIFICATION_V3_20261011
# BOOKING_REFERENCE_MANAGEMENT_V2_20261011
# SAME_CALL_NAME_CORRECTION_V1_20261011
# CONFIRMATION_AND_SLOT_RECOVERY_V1_20261010
# NATURAL_HANDOVER_RELATIONSHIP_V1_20261004
# BOOKING_CALLER_REASON_V1_20261003
# ALL_OPERATIONS_NAME_SPELLING_V1_20261003
# WHOLE_CALL_SPEAKER_EVIDENCE_V1_20261003
# CONTROLLED_NAMEFLOW_V3_20261003
import os
import json
import time

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client


# ============================================================
# BASIC SETUP
# ============================================================

load_dotenv()

SYDNEY_TZ = ZoneInfo("Australia/Sydney")

# QWEN_LATENCY_PAYLOAD_OPTIMIZATION_20260903
QWEN_MODEL = "qwen3.7-plus"
MAX_AVAILABILITY_ROWS_PER_QUERY = 40


def sydney_now():

    return datetime.now(
        SYDNEY_TZ
    )


def slot_is_in_future(
    slot
):

    appointment_date = str(
        slot.get(
            "appointment_date",
            ""
        )
    ).strip()

    appointment_time = str(
        slot.get(
            "appointment_time",
            ""
        )
    ).strip()

    if (
        not appointment_date
        or not appointment_time
    ):

        return False

    normalized_time = (
        appointment_time.replace(
            "Z",
            "+00:00"
        )
    )

    try:

        slot_datetime = (
            datetime.fromisoformat(
                appointment_date
                + "T"
                + normalized_time
            )
        )

    except ValueError:

        return False

    if slot_datetime.tzinfo is None:

        slot_datetime = (
            slot_datetime.replace(
                tzinfo=SYDNEY_TZ
            )
        )

    else:

        slot_datetime = (
            slot_datetime.astimezone(
                SYDNEY_TZ
            )
        )

    return (
        slot_datetime
        > sydney_now()
    )


qwen = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    timeout=30.0,
    max_retries=0,
)

supabase_key = (
    os.getenv("SUPABASE_SECRET_KEY")
    or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
)

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    supabase_key,
)


# ============================================================
# LOAD ACTIVE CLINIC SERVICES ONCE
# ============================================================

service_response = (
    supabase
    .table("services")
    .select(
        "service_id,"
        "service_name,"
        "department,"
        "description"
    )
    .eq("active", True)
    .order("service_id")
    .execute()
)

services = service_response.data or []

service_by_name = {
    service["service_name"]: service
    for service in services
}

service_by_id = {
    service["service_id"]: service
    for service in services
}

service_knowledge = json.dumps(
    [
        {
            "service_name":
                service["service_name"],

            "department":
                service["department"],

            "description":
                service["description"],
        }
        for service in services
    ],
    indent=2,
)


# ============================================================
# LOAD VERIFIED ADMIN / RAG KNOWLEDGE ONCE
# ============================================================

knowledge_response = (
    supabase
    .table("clinic_knowledge")
    .select(
        "knowledge_id,"
        "category,"
        "title,"
        "content"
    )
    .eq("active", True)
    .order("knowledge_id")
    .execute()
)

knowledge_rows = (
    knowledge_response.data or []
)


# ============================================================
# HELPERS
# ============================================================

def format_date(date_text):

    date_object = datetime.strptime(
        date_text,
        "%Y-%m-%d"
    )

    return date_object.strftime(
        "%A, %d %B %Y"
    )


def format_time(time_text):

    time_object = datetime.strptime(
        time_text,
        "%H:%M:%S"
    )

    return time_object.strftime(
        "%I:%M %p"
    ).lstrip("0")


# ============================================================
# TOOL DEFINITIONS GIVEN TO QWEN
# ============================================================

TOOLS = [

    {
        "type": "function",

        "function": {

            "name":
                "check_service_availability",

            "description":
                "Check the real clinic database for actual "
                "providers and currently available appointment "
                "slots for an available clinic service. "
                "Use for appointment availability and when a "
                "clinical question clearly maps to an available "
                "clinic service. Never invent availability.",

            "parameters": {

                "type": "object",

                "properties": {

                    "service_name": {
                        "type": "string",
                        "description":
                            "Exact available clinic service name."
                    },

                    "consultant_gender": {
                        "type": "string",
                        "enum": ["female", "male", "no_preference"],
                        "description":
                            "The patient's stated consultant preference. "
                            "Never infer this from voice, name or gender."
                    },

                    "consultation_mode": {
                        "type": "string",
                        "enum": ["in_person", "video", "audio"],
                        "description":
                            "Required for Medical Certificate Consultation. "
                            "Video and audio are online options."
                    }
                },

                "required": [
                    "service_name"
                ]
            }
        }
    },

    {
        "type": "function",

        "function": {

            "name":
                "book_appointment",

            "description":
                "Verify the patient's full-name spelling, then "
                "prepare or execute a booking for one real "
                "previously offered slot. The first call with a new "
                "patient name requests letter-by-letter spelling "
                "and MUST NOT write to the database. After the "
                "patient spells the full name in a later turn, call "
                "this tool with the corrected name and the same slot "
                "to prepare the booking. Only after the patient "
                "explicitly confirms the exact prepared booking in "
                "another later turn may identical arguments execute "
                "the booking.",

            "parameters": {

                "type": "object",

                "properties": {

                    "patient_name": {
                        "type": "string"
                    },

                    "slot_id": {
                        "type": "integer"
                    },

                    "consultation_mode": {
                        "type": "string",
                        "enum": ["in_person", "video", "audio"]
                    },

                    "insurance_status": {
                        "type": "string",
                        "enum": ["insured", "not_insured"]
                    },

                    "certificate_purpose": {
                        "type": "string"
                    },

                    "requested_date_range": {
                        "type": "string"
                    }
                },

                "required": [
                    "patient_name",
                    "slot_id"
                ]
            }
        }
    },

    {
        "type": "function",

        "function": {

            "name":
                "find_booked_appointments",

            "description":
                "Find the patient's real currently booked "
                "appointments. Use for appointment lookup, "
                "rescheduling, and cancellation.",

            "parameters": {

                "type": "object",

                "properties": {

                    "patient_name": {
                        "type": "string"
                    }
                },

                "required": [
                    "patient_name"
                ]
            }
        }
    },

    {
        "type": "function",

        "function": {

            "name":
                "reschedule_appointment",

            "description":
                "Prepare or execute rescheduling of a validated "
                "appointment to one real previously offered slot. "
                "The first call prepares the change only. After "
                "the patient explicitly confirms the exact old "
                "and new appointment details in a later turn, "
                "call this tool again with identical arguments.",

            "parameters": {

                "type": "object",

                "properties": {

                    "appointment_id": {
                        "type": "integer"
                    },

                    "new_slot_id": {
                        "type": "integer"
                    }
                },

                "required": [
                    "appointment_id",
                    "new_slot_id"
                ]
            }
        }
    },

    {
        "type": "function",

        "function": {

            "name":
                "cancel_appointment",

            "description":
                "Prepare or execute cancellation of a validated "
                "currently booked appointment. The first call "
                "prepares cancellation only. After the patient "
                "explicitly confirms cancellation of those exact "
                "appointment details in a later turn, call this "
                "tool again with the identical appointment_id.",

            "parameters": {

                "type": "object",

                "properties": {

                    "appointment_id": {
                        "type": "integer"
                    }
                },

                "required": [
                    "appointment_id"
                ]
            }
        }
    },

    {
        "type": "function",

        "function": {

            "name":
                "get_admin_information",

            "description":
                "Retrieve verified non-clinical clinic information "
                "for opening hours, location, services, policies, "
                "documents, booking procedure, cancellation, "
                "rescheduling and other administrative questions.",

            "parameters": {

                "type": "object",

                "properties": {

                    "question": {
                        "type": "string"
                    }
                },

                "required": [
                    "question"
                ]
            }
        }
    }
]

for _booking_tool in TOOLS:
    if _booking_tool['function']['name'] == 'book_appointment':
        _booking_tool['function']['parameters']['properties']['reason_for_visit'] = {
            'type': 'string',
            'description': 'Optional exact verbatim quote of the reason the patient gave for this appointment. Never infer a diagnosis or invent a reason. Omit when not given.'}



# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are a highly professional, formal, responsible and courteous
AI clinic receptionist.

Communicate as a trained healthcare reception professional:
calm, respectful, patient, concise and attentive.

This is ONE CONTINUOUS multi-turn conversation.

Remember and use relevant information from previous turns.

Always understand the patient's newest complete statement in the
context of the conversation before deciding how to respond.

If the patient changes their mind, corrects information, changes
topic, or interrupts a previous response, treat the newest clear
instruction as the current conversational direction.

Never continue acting on an older request when the patient has
clearly replaced or corrected it.

CURRENT ACTIVE CLINIC SERVICES:

{service_knowledge}


============================================================
CLINICAL SAFETY
============================================================

1. Never diagnose.

2. Never recommend medication.

3. Never recommend treatment.

4. Never provide dosage advice.

5. Never advise starting, stopping, changing,
   or continuing medication or treatment.

6. Never directly answer treatment, medication,
   diagnosis, dosage, or other clinical-advice questions.

7. You may identify an appropriate clinic SERVICE.
   Service routing is not diagnosis.

8. If a concern clearly matches an available specialist,
   use that specialist.

9. When the correct specialist exists,
   do not additionally offer General Practice.

10. For a clinical question that clearly maps to an
    available service, call check_service_availability.

11. Briefly explain that you cannot provide medical,
    medication, treatment, or clinical advice, then offer
    the real appropriate provider and real slots.

12. If the required specialist is NOT available,
    state clearly that the service is unavailable.

13. Never invent a specialist.

14. General Practice may be offered as an initial
    assessment/referral option only when the specialist
    is unavailable, or when GP itself is the appropriate service.


============================================================
ADMIN / RAG
============================================================

15. Administrative clinic information must come from
    get_admin_information.

16. Never invent administrative clinic facts.

17. Never use admin/RAG information to provide clinical,
    treatment, medication, or dosage advice.


============================================================
GENERAL CONVERSATION
============================================================

18. Greetings, thanks, harmless jokes, casual questions and
    unrelated non-clinical comments may be answered briefly.

19. If the patient asks a random or off-topic question,
    remain calm and professional. Do not become confused and
    do not invent clinic information.

20. If an off-topic request is outside the receptionist role,
    politely state the limitation in one short sentence.
    Do not lecture the patient.

21. Do not force the conversation back to appointment booking
    after every unrelated comment.

22. If an active clinic task existed before an off-topic comment,
    preserve that context and return to it only when natural or
    when the patient refers to it again.

23. Do not pretend to be human.

24. Do not claim personal relationships, feelings,
    romantic feelings, favourites, memories, hobbies,
    family, personal experiences, or a personal life.

25. Do not emotionally reciprocate romantic comments.

26. Never respond defensively or argumentatively to rude,
    unusual, confusing or unexpected comments.

27. If the patient's intended request is genuinely unclear,
    ask one short, natural clarification question instead of
    guessing.


============================================================
INTERRUPTION / CORRECTION BEHAVIOUR
============================================================

When the patient interrupts, corrects themselves, changes their
mind, or replaces a previous request:

- Give priority to the patient's newest complete statement.
- Preserve relevant earlier context.
- Do not answer a partial fragment as if it were a complete turn.
- Do not continue the abandoned request unless the patient
  explicitly returns to it.
- If the new statement is understandable, respond directly
  without asking the patient to repeat everything.
- If only one important detail is unclear, clarify only that
  detail.
- Never criticize or mention that the patient interrupted.
- Respond naturally as though a professional receptionist
  patiently listened to the correction.

Example conversational meaning:

Previous request:
"I need a neurologist appointment."

New patient statement:
"Actually, I need a cardiologist instead."

Interpretation:
The cardiology request replaces the neurology request unless
the patient later says otherwise.



============================================================
MULTI-TURN MEMORY
============================================================

25. Maintain context throughout this session.

26. Do not ask the patient to repeat information already
    provided in this session.

27. Understand references such as:
    "that",
    "the first one",
    "the Friday one",
    "the earliest one",
    "cancel that",
    "move that appointment",
    "the one I just booked".

28. Ask only for genuinely missing information.

29. If important information is ambiguous,
    ask a natural clarification question.


============================================================
DATABASE / DATE SAFETY
============================================================

30. Never invent providers.

31. Never invent appointment availability.

32. Never invent appointment weekdays, dates or times.

33. Repeat display_date and display_time exactly as
    returned by database tools.

34. Never expose appointment_id, slot_id,
    doctor_id or other internal database IDs.


============================================================
BOOKING
============================================================

35. Real availability comes from check_service_availability.

36. A booking requires:
    - one real offered slot,
    - patient full name,
    - verified letter-by-letter spelling of that full name,
    - explicit confirmation.

37. Ask only for missing information.

37A. Before naming doctors, ask once whether the patient prefers a
     female consultant, a male consultant, or has no preference.
     Never infer this preference from the patient's voice, name or gender.

37B. For Medical Certificate Consultation, explain that the appointment
     is a clinician assessment and a certificate is not guaranteed. Ask
     whether the appointment is in person or online; for online, ask video
     or audio. Also collect insurance status, purpose and requested date
     range. The fictional prototype fee is AUD 0 with insurance and AUD 45
     without insurance. Never claim that a real payment is being processed.

38. Once the patient name and one exact real slot are known,
    call book_appointment. If the tool returns
    requires_name_spelling=true, ask the patient to spell the
    complete full name letter by letter once. Preserve correctly
    heard name parts. If the tool identifies only one missing part,
    ask only for that part and never restart the whole spelling.

39. On a LATER turn containing spelling, reconstruct the name and call
    book_appointment with the same real slot. If the tool asks for a
    read-back confirmation, read the complete candidate once. On the
    next clear confirmation such as "yes, correct", call the tool again.
    Never ask for spelling again after the exact candidate was captured.

40. After the verified-name call returns PREPARE details, read back
    the exact corrected patient name, provider, date and time from
    confirmation_details and ask once for explicit confirmation.
    Only on another LATER patient turn with clear confirmation may
    book_appointment be called again with identical arguments.
    Never claim booking succeeded unless success=true.


============================================================
RESCHEDULING
============================================================

41. Identify the booked appointment first.

42. Offer only real alternative slots for the same service.

43. Once the real old appointment and one exact real new
    option are known, call reschedule_appointment immediately
    to PREPARE. Do not ask a separate approval question first.

44. Call reschedule_appointment once to PREPARE the change.
    Use its confirmation_details for the single confirmation
    request. After the patient explicitly confirms those exact
    details in a LATER turn, call reschedule_appointment again
    using identical arguments.

45. Never say rescheduled unless success=true.


============================================================
CANCELLATION
============================================================

46. Identify the booked appointment first.

47. Identify the exact real appointment, then call
    cancel_appointment immediately to PREPARE cancellation.
    Do not ask a separate approval question before PREPARE.

48. Call cancel_appointment once to PREPARE cancellation.
    Use its confirmation_details for the single confirmation
    request. After the patient explicitly confirms cancellation
    of that exact appointment in a LATER turn, call
    cancel_appointment again with the identical argument.

49. Never say cancelled unless success=true.


============================================================
CURRENT AVAILABILITY AND PRIVACY
============================================================

49A. Never claim a slot is currently available using only an
     earlier message or cached list after any booking,
     rescheduling or cancellation attempt. Refresh availability
     with check_service_availability before offering it again.

49B. When the patient says they are a different person or are
     booking for someone else, do not reuse the previous
     patient's identity, appointment or selected slot.

49C. Never reveal, guess or discuss another patient's identity,
     appointment details or conversation session. If a slot is
     unavailable, say only that it is no longer available and
     offer freshly retrieved alternatives.

49D. Never invent explanations such as a booking belonging to a
     different patient or separate session.


============================================================
CLINIC KNOWLEDGE AND SCOPE
============================================================

49E. Use get_admin_information for questions about clinic
     facilities, amenities, opening hours, location, contact,
     documents, policies and other general clinic information.

49F. Answer clinic-information questions only from verified
     clinic records. Never invent a facility, address, telephone
     number, service, policy or opening time.

49G. If the patient discusses a genuinely unrelated topic, first
     acknowledge it naturally and briefly. Explain that your role
     is the clinic receptionist and that you can help with clinic
     bookings, administration and general clinic enquiries. Do
     not repeat a robotic stock sentence.

49H. Do not provide medication, diagnosis or treatment advice.
     Explain that clinical guidance must come from a qualified
     clinician, while still helping with appropriate clinic
     administration or scheduling.

============================================================
STYLE
============================================================

50. Use a formal, respectful, responsible healthcare
    receptionist tone.

51. Be warm and natural, but not overly casual.

52. Keep spoken responses concise so the patient does not
    have to wait through unnecessarily long answers.

53. Prefer short, clear sentences that sound natural when
    spoken aloud.

54. Do not sound robotic, theatrical or overly enthusiastic.

55. Do not unnecessarily repeat information already understood.

56. Do not repeatedly push appointment booking.

57. Let the patient naturally change topics and return later.

58. Allow the patient to finish expressing their request before
    responding.

59. When clarification is required, ask only one focused
    question at a time.

60. For confirmed information, speak confidently but never
    claim anything that has not been verified through the
    appropriate clinic tool or database.

61. When a database or tool check is required, never guess in
    order to respond faster.

62. Prefer a useful concise answer over a long explanation,
    especially during live voice conversation.

63. Produce plain spoken text only. Never use Markdown, asterisks,
    bullet points, numbered lists, headings, tables or emojis.

64. By default, answer in no more than two short sentences and
    no more than 45 words.

65. If several services or choices exist, mention only the most
    relevant options, then ask one focused follow-up question.

66. Give a longer explanation only when the patient explicitly
    asks for more detail.

67. Do not repeat the patient's complete statement. Respond
    naturally and directly.

68. Answer the patient's latest question directly. When the answer is
    complete, stop without adding a routine appointment invitation.

69. Offer appointment booking or availability only when the patient's
    latest message explicitly asks to book, schedule, check availability,
    or clearly accepts an earlier booking offer.

70. A question about symptoms, services, fees, policies or clinic
    information alone is not booking intent. Do not ask whether the
    patient wants an appointment after answering such a question.

71. When requested information is absent from the verified clinic
    records, state this politely and briefly. Then identify the most
    appropriate team or qualified professional and give the verified
    next step for obtaining the information.

72. Route missing-information enquiries appropriately:
    fees, payments or insurance to billing or administration staff;
    medical advice, symptoms or clinical uncertainty to a GP or
    qualified clinical team; prescriptions, referrals and test results
    to the treating clinician or clinical team; records and privacy
    matters to records, privacy or administration staff; complaints to
    the practice manager; and unresolved scheduling matters to
    scheduling or administration staff.

73. You are the clinic's AI receptionist. Do not tell the patient merely
    to "contact reception," because that is circular. Name the relevant
    team or professional and explain the next available contact route.

74. Give a telephone number, email address, location, opening time or
    transfer option only when it has been verified through clinic data
    or an available tool. Never invent contact information.

75. Never claim that you have called, messaged, transferred, notified or
    created a request for staff unless a real connected tool completed
    that action.

76. Never use phrases such as "I am not here to tell you that" or
    "I cannot tell you that." Use calm, respectful and helpful wording.

77. Ask a follow-up question only when information is genuinely needed
    to complete the patient's current request. Avoid repetitive closing
    questions and routine appointment offers.
"""

SYSTEM_PROMPT += """
When preparing a NEW booking, include reason_for_visit only if the caller stated
a reason for that appointment. Copy a short exact quote from a patient turn in
the current patient context. Never turn symptoms into a diagnosis, copy assistant
suggestions, or carry a previous patient's reason into a new request. Do not ask
an extra reason question if the caller did not provide one; it is optional.
Read back the server's exact prepared details, including any stated reason.
Reason capture does not replace name verification or final booking confirmation.
"""


# ============================================================
# ONE RECEPTIONIST SESSION
# ============================================================

class ReceptionistSession:

    def __init__(self):

        current_sydney = (
            sydney_now()
        )

        self.session_state = {
            "patient_name": None,
            "active_service": None,
            "last_available_slots": [],
            "selected_slot_id": None,
            "last_found_appointments": [],
            "active_appointment_id": None,
            "pending_action": None,
            "pending_name_verification": None,
            "verified_patient_name": None,
            "verified_patient_name_key": None,
            "observed_spelled_name_key": None,
            "observed_spelled_name_confirmed": False,
            "observed_spelled_name_parts": [],
            "name_spelling_mismatch_count": 0,
            "last_name_evidence_text": None,
            "speaker_change_pending": None,
            "returning_voice_candidate_pending": None,
        }

        self.current_patient_text = ""

        self.messages = [
            {
                "role": "system",
                "content": (
                    SYSTEM_PROMPT
                    + "\n\nLIVE SYDNEY DATE AND TIME: "
                    + current_sydney.strftime(
                        "%A, %d %B %Y at %I:%M %p"
                    )
                    + " Australia/Sydney."
                    + "\nInterpret today, tomorrow and other "
                    + "relative dates from this live clock. "
                    + "Never offer or transact an appointment "
                    + "whose date and time have already passed."
                ),
            }
        ]


    # ========================================================
    # REAL TOOL: AVAILABILITY
    # ========================================================

    def check_service_availability(
        self,
        service_name,
        consultant_gender=None,
        consultation_mode=None,
    ):

        service = service_by_name.get(service_name)
        if not service:
            return {
                "success": False,
                "reason": "Service is not available at this clinic.",
                "service_name": service_name,
            }

        allowed_genders = {"female", "male", "no_preference"}
        allowed_modes = {"in_person", "video", "audio"}
        normalized_gender = str(consultant_gender or "").strip().lower()
        normalized_mode = str(consultation_mode or "").strip().lower()

        missing_selections = []
        if normalized_gender not in allowed_genders:
            missing_selections.append({
                "field": "consultant_gender",
                "question": (
                    "Do you prefer a female consultant, a male consultant, "
                    "or do you have no preference?"
                ),
                "choices": ["female", "male", "no_preference"],
            })
        if (
            service_name == "Medical Certificate Consultation"
            and normalized_mode not in allowed_modes
        ):
            missing_selections.append({
                "field": "consultation_mode",
                "question": (
                    "Would you like an in-person appointment or an online "
                    "appointment by video or audio?"
                ),
                "choices": ["in_person", "video", "audio"],
            })

        if missing_selections:
            return {
                "success": True,
                "service_name": service_name,
                "needs_clarification": True,
                "required_selections": missing_selections,
                "instruction": (
                    "Ask only the listed questions. Do not name consultants "
                    "or invent availability before the patient answers."
                ),
            }

        if service_name != "Medical Certificate Consultation":
            normalized_mode = "in_person"

        self.session_state['availability_preferences_v1'] = dict(service_name=service_name, consultant_gender=normalized_gender, consultation_mode=normalized_mode)
        service_id = service["service_id"]
        provider_response = (
            supabase.table("doctor_services")
            .select(
                "doctor_id,doctors(doctor_id,doctor_name,specialty,consultant_gender)"
            )
            .eq("service_id", service_id)
            .execute()
        )

        providers = []
        for row in provider_response.data or []:
            doctor = row.get("doctors")
            if not doctor:
                continue
            doctor_gender = str(doctor.get("consultant_gender") or "").lower()
            if normalized_gender != "no_preference" and doctor_gender != normalized_gender:
                continue
            providers.append(doctor)

        provider_by_id = {
            provider["doctor_id"]: provider
            for provider in providers
        }
        provider_ids = list(provider_by_id)

        if not provider_ids:
            return {
                "success": True,
                "service_name": service_name,
                "consultant_gender": normalized_gender,
                "consultation_mode": normalized_mode,
                "providers": [],
                "available_slots": [],
                "reason": "No consultant currently matches the stated preference.",
                "instruction": (
                    "State that the preferred consultant is unavailable and ask "
                    "whether another available consultant is acceptable."
                ),
            }

        slot_query = (
            supabase.table("slots")
            .select(
                "slot_id,doctor_id,service_id,appointment_date,"
                "appointment_time,available,consultation_mode"
            )
            .eq("service_id", service_id)
            .eq("available", True)
            .in_("doctor_id", provider_ids)
            .eq("consultation_mode", normalized_mode)
            .gte("appointment_date", sydney_now().date().isoformat())
            .order("appointment_date")
            .order("appointment_time")
            .limit(MAX_AVAILABILITY_ROWS_PER_QUERY)
        )
        slot_response = slot_query.execute()

        available_slots = []
        for slot in slot_response.data or []:
            if not slot_is_in_future(slot):
                continue
            provider = provider_by_id.get(slot["doctor_id"], {})
            available_slots.append({
                "slot_id": slot["slot_id"],
                "doctor_id": slot["doctor_id"],
                "provider": provider.get("doctor_name", "Unknown provider"),
                "consultant_gender": provider.get("consultant_gender"),
                "specialty": provider.get("specialty"),
                "consultation_mode": slot.get("consultation_mode", normalized_mode),
                "service_name": service_name,
                "appointment_date": slot["appointment_date"],
                "appointment_time": slot["appointment_time"],
                "display_date": format_date(slot["appointment_date"]),
                "display_time": format_time(slot["appointment_time"]),
            })

        self.session_state["active_service"] = service_name
        self.session_state["last_available_slots"] = available_slots
        return {
            "success": True,
            "service_name": service_name,
            "consultant_gender": normalized_gender,
            "consultation_mode": normalized_mode,
            "providers": providers,
            "available_slots": available_slots,
        }


    # VERIFIED_AVAILABILITY_FAST_RESPONSE_20260906
    # Convert already-verified slot data to a short patient response.
    # Qwen still performs natural-language understanding and tool selection.
    # This removes only the second Qwen wording call for simple availability.

    def _fast_availability_response(
        self,
        tool_result
    ):

        if not isinstance(tool_result, dict):
            return None

        if not tool_result.get("success"):
            return None

        if tool_result.get("needs_clarification"):
            return None

        available_slots = (
            tool_result.get("available_slots")
            or []
        )

        service_name = str(
            tool_result.get("service_name")
            or "the requested service"
        )

        patient_text = str(
            self.current_patient_text or ""
        ).lower()

        today = sydney_now().date()
        target_date = None
        date_description = None

        if "tomorrow" in patient_text:
            target_date = today + timedelta(days=1)
            date_description = "tomorrow, " + format_date(
                target_date.isoformat()
            )

        elif "today" in patient_text:
            target_date = today
            date_description = "today, " + format_date(
                target_date.isoformat()
            )

        else:
            explicit_date_words = (
                "monday", "tuesday", "wednesday",
                "thursday", "friday", "saturday",
                "sunday", "next week", "this week"
            )

            if any(
                date_word in patient_text
                for date_word in explicit_date_words
            ):
                return None

        if target_date is not None:
            target_iso = target_date.isoformat()
            matching_slots = [
                slot
                for slot in available_slots
                if str(slot.get("appointment_date"))
                == target_iso
            ]

        else:
            matching_slots = []

            if available_slots:
                earliest_date = str(
                    available_slots[0]["appointment_date"]
                )
                matching_slots = [
                    slot
                    for slot in available_slots
                    if str(slot.get("appointment_date"))
                    == earliest_date
                ]
                date_description = format_date(
                    earliest_date
                )

        # AVAILABILITY_SAME_DATE_DISPLAY_20260906
        # Never present two rows that look identical to the patient.
        offered_slots = []
        offered_display_keys = set()

        for slot in matching_slots:
            display_key = (
                str(slot.get("provider") or "").strip().lower(),
                str(slot.get("appointment_date") or ""),
                str(slot.get("display_time") or "").strip().lower(),
            )

            if display_key in offered_display_keys:
                continue

            offered_display_keys.add(display_key)
            offered_slots.append(slot)

            if len(offered_slots) >= 2:
                break

        if offered_slots:
            slot_lines = []

            for slot in offered_slots:
                slot_lines.append(
                    "- "
                    + str(slot.get("provider") or "Provider")
                    + " at "
                    + str(slot.get("display_time") or "")
                )

            return (
                "Yes, there are "
                + service_name
                + " appointments available "
                + str(date_description)
                + ". The available slots are:\n\n"
                + "\n".join(slot_lines)
                + "\n\nWould you like to book one of these "
                + "appointments? If so, may I have your full "
                + "name please?"
            )

        if target_date is not None:
            next_slots = available_slots[:2]

            if next_slots:
                next_lines = []

                for slot in next_slots:
                    next_lines.append(
                        "- "
                        + str(slot.get("provider") or "Provider")
                        + " on "
                        + str(slot.get("display_date") or "")
                        + " at "
                        + str(slot.get("display_time") or "")
                    )

                return (
                    "There are no "
                    + service_name
                    + " appointments available "
                    + str(date_description)
                    + ". The next available options are:\n\n"
                    + "\n".join(next_lines)
                    + "\n\nWould you like one of these options?"
                )

            return (
                "There are no "
                + service_name
                + " appointments currently available."
            )

        return None


    # ========================================================
    # PATIENT NAME SPELLING VERIFICATION GATE
    # ========================================================

    def _name_letters(
        self,
        value
    ):

        return "".join(
            character.lower()
            for character in str(
                value or ""
            )
            if character.isalpha()
        )


    def _spoken_spelling_letters(
        self,
        patient_text
    ):

        normalized = "".join(
            character.lower()
            if character.isalpha()
            else " "
            for character in str(
                patient_text or ""
            )
        )

        tokens = normalized.split()

        spelling_runs = []
        current_run = []

        for token in tokens:

            if len(token) == 1:

                current_run.append(
                    token
                )

                continue


            if len(current_run) >= 2:

                spelling_runs.extend(
                    current_run
                )

            current_run = []


        if len(current_run) >= 2:

            spelling_runs.extend(
                current_run
            )


        isolated_letters = "".join(spelling_runs)

        if isolated_letters:
            return isolated_letters

        pending = self.session_state.get("pending_name_verification")
        spelling_cues = (
            "spell",
            "spelling",
            "letter",
            "as in",
            "double",
        )
        context_allows_word_letters = (
            isinstance(pending, dict)
            or any(cue in normalized for cue in spelling_cues)
        )
        if not context_allows_word_letters:
            return ""

        nato_letters = {
            "alpha": "a", "bravo": "b", "charlie": "c",
            "delta": "d", "echo": "e", "foxtrot": "f",
            "golf": "g", "hotel": "h", "india": "i",
            "juliett": "j", "juliet": "j", "kilo": "k",
            "lima": "l", "mike": "m", "november": "n",
            "oscar": "o", "papa": "p", "quebec": "q",
            "romeo": "r", "sierra": "s", "tango": "t",
            "uniform": "u", "victor": "v", "whiskey": "w",
            "xray": "x", "yankee": "y", "zulu": "z",
        }
        spoken_letter_names = {
            "ay": "a", "bee": "b", "cee": "c", "see": "c",
            "dee": "d", "ee": "e", "ef": "f", "gee": "g",
            "aitch": "h", "eye": "i", "jay": "j", "kay": "k",
            "el": "l", "em": "m", "en": "n", "oh": "o",
            "pee": "p", "cue": "q", "queue": "q", "are": "r",
            "ess": "s", "tee": "t", "you": "u", "vee": "v",
            "ex": "x", "why": "y", "zee": "z", "zed": "z",
        }

        expected_key = ""
        if isinstance(pending, dict):
            expected_key = self._name_letters(pending.get("patient_name"))

        candidate = []
        reliable_word_evidence = False
        index = 0
        while index < len(tokens):
            token = tokens[index]

            if token == "double" and index + 1 < len(tokens):
                following = tokens[index + 1]
                letter = (
                    following if len(following) == 1
                    else nato_letters.get(following)
                    or spoken_letter_names.get(following)
                )
                if letter:
                    candidate.extend([letter, letter])
                    reliable_word_evidence = True
                    index += 2
                    continue

            if (
                len(token) == 1
                and index + 2 < len(tokens)
                and tokens[index + 1:index + 3] == ["as", "in"]
            ):
                candidate.append(token)
                reliable_word_evidence = True
                index += 4
                continue

            if token in nato_letters:
                candidate.append(nato_letters[token])
                reliable_word_evidence = True
            elif len(token) == 1:
                candidate.append(token)
            elif token in spoken_letter_names:
                candidate.append(spoken_letter_names[token])

            index += 1

        word_letters = "".join(candidate)
        matches_expected_part = (
            not expected_key
            or word_letters in expected_key
        )
        if (
            len(word_letters) >= 2
            and matches_expected_part
            and (reliable_word_evidence or len(word_letters) >= 3)
        ):
            return word_letters

        return ""


    # PROACTIVE_NAME_SPELLING_CONFIRMATION_20260906
    # Capture an explicitly stated, letter-by-letter full name as soon as
    # it is spoken, rather than waiting for the later booking tool call.

    def _capture_proactive_name_spelling_before_nameflow(
        self,
        patient_text
    ):

        spoken_letters = self._spoken_spelling_letters(
            patient_text
        )

        normalized = self._normalize_confirmation_text(
            patient_text
        )

        padded = " " + normalized + " "

        full_name_phrases = (
            " my full name ",
            " my complete full name ",
            " it is my full name ",
            " it s my full name ",
            " it is my complete full name ",
            " it s my complete full name ",
            " that is my full name ",
            " that s my full name ",
            " the complete name is correct ",
            " the name is correct ",
            " it is correct ",
            " it s correct ",
            " yes it is correct ",
            " yes it s correct ",
            " yes correct ",
            " correct ",
            " that is correct ",
            " thats correct ",
        )

        explicitly_confirmed = any(
            phrase in padded
            for phrase in full_name_phrases
        )

        pending = self.session_state.get(
            "pending_name_verification"
        )

        expected_key = ""

        if isinstance(pending, dict):
            expected_key = self._name_letters(
                pending.get("patient_name")
            )

        if (
            expected_key
            and self.session_state.get(
                "observed_spelled_name_key"
            ) == expected_key
            and normalized in {
                "yes",
                "yeah",
                "yep",
                "correct",
                "yes correct",
                "that is correct",
                "it is correct",
            }
        ):
            explicitly_confirmed = True

        if self._is_explicit_rejection(patient_text):
            explicitly_confirmed = False
        if len(spoken_letters) >= 2:
            parts = list(
                self.session_state.get(
                    "observed_spelled_name_parts"
                ) or []
            )

            if expected_key:
                if spoken_letters == expected_key:
                    parts = [spoken_letters]
                elif spoken_letters in expected_key:
                    if spoken_letters not in parts:
                        parts.append(spoken_letters)
                    parts.sort(
                        key=lambda value: expected_key.find(value)
                    )
                else:
                    self.session_state[
                        "last_unmatched_spelling_key"
                    ] = spoken_letters
            else:
                parts = [spoken_letters]

            combined = "".join(parts)

            self.session_state[
                "observed_spelled_name_parts"
            ] = parts

            if expected_key and combined == expected_key:
                self.session_state[
                    "observed_spelled_name_key"
                ] = expected_key
            elif not expected_key:
                self.session_state[
                    "observed_spelled_name_key"
                ] = spoken_letters

            if (
                explicitly_confirmed
                and self.session_state.get(
                    "observed_spelled_name_key"
                ) == expected_key
            ):
                self.session_state[
                    "observed_spelled_name_confirmed"
                ] = True

            print(
                "NAME SPELLING EVIDENCE: captured parts",
                parts,
                "| Exact candidate:",
                combined == expected_key if expected_key else False,
            )

            return

        if (
            self.session_state.get(
                "observed_spelled_name_key"
            )
            and explicitly_confirmed
            and (
                not expected_key
                or self.session_state.get(
                    "observed_spelled_name_key"
                ) == expected_key
            )
        ):
            self.session_state[
                "observed_spelled_name_confirmed"
            ] = True

            print(
                "NAME SPELLING EVIDENCE: prior spelling confirmed"
            )


    def _check_patient_name_spelling_before_nameflow(
        self,
        patient_name,
        slot_id
    ):

        clean_name = " ".join(
            str(
                patient_name or ""
            ).split()
        )

        name_key = self._name_letters(
            clean_name
        )

        if not clean_name or len(name_key) < 2:

            return {
                "success": False,
                "requires_name_spelling": True,
                "reason":
                    "A complete patient name is required before "
                    "booking preparation.",
                "instruction":
                    "Ask for the patient's complete full name."
            }


        verified_key = self.session_state.get(
            "verified_patient_name_key"
        )

        if verified_key == name_key:

            return None

        observed_spelled_key = self.session_state.get(
            "observed_spelled_name_key"
        )

        if (
            self.session_state.get(
                "observed_spelled_name_confirmed"
            )
            and observed_spelled_key == name_key
        ):

            self.session_state[
                "pending_name_verification"
            ] = None

            self.session_state[
                "verified_patient_name"
            ] = clean_name

            self.session_state[
                "verified_patient_name_key"
            ] = name_key

            print(
                "NAME SPELLING VERIFICATION: prior explicit "
                "spelling accepted"
            )

            return None


        pending = self.session_state.get(
            "pending_name_verification"
        )

        spoken_letters = self._spoken_spelling_letters(
            self.current_patient_text
        )

        same_pending_slot = (
            isinstance(
                pending,
                dict
            )
            and pending.get(
                "slot_id"
            ) == slot_id
        )

        if observed_spelled_key == name_key:
            self.session_state["pending_name_verification"] = {"patient_name": clean_name, "slot_id": slot_id}
            return {
                "success": False,
                "requires_name_spelling": True,
                "requires_name_readback_confirmation": True,
                "spelling_mismatch": False,
                "provisional_patient_name": clean_name,
                "reason": "The exact spelling was captured and now needs one read-back confirmation.",
                "instruction": (
                    f"Read back the complete name {clean_name} once and ask if it is correct. "
                    "Accept a clear yes or correct on the next turn. Do not ask for spelling again."
                ),
            }


        self.session_state[
            "pending_name_verification"
        ] = {
            "patient_name": clean_name,
            "slot_id": slot_id,
        }

        parts = list(
            self.session_state.get(
                "observed_spelled_name_parts"
            ) or []
        )

        combined_parts = "".join(parts)

        if parts and name_key.startswith(combined_parts):
            name_components = clean_name.split()
            component_keys = [
                self._name_letters(component)
                for component in name_components
            ]
            captured_length = len(combined_parts)
            traversed = 0
            missing_components = []

            for component, component_key in zip(
                name_components,
                component_keys
            ):
                traversed += len(component_key)
                if traversed > captured_length:
                    missing_components.append(component)

            missing_label = " ".join(missing_components).strip()

            if missing_label:
                return {
                    "success": False,
                    "requires_name_spelling": True,
                    "spelling_mismatch": False,
                    "provisional_patient_name": clean_name,
                    "captured_name_parts": parts,
                    "missing_name_part": missing_label,
                    "reason": "A correct part of the proposed name has already been captured.",
                    "instruction": (
                        f"Keep the spelling already captured. Ask only for {missing_label} "
                        "letter by letter. Do not ask for the complete name again."
                    ),
                }

        if (
            same_pending_slot
            and spoken_letters
            and spoken_letters != name_key
        ):

            evidence_text = self.current_patient_text.strip().lower()

            if self.session_state.get(
                "last_name_evidence_text"
            ) != evidence_text:
                self.session_state[
                    "last_name_evidence_text"
                ] = evidence_text
                self.session_state[
                    "name_spelling_mismatch_count"
                ] = int(
                    self.session_state.get(
                        "name_spelling_mismatch_count"
                    ) or 0
                ) + 1

            mismatch_count = int(
                self.session_state.get(
                    "name_spelling_mismatch_count"
                ) or 0
            )

            if mismatch_count >= 2:
                return {
                    "success": False,
                    "requires_name_spelling": False,
                    "requires_human_assistance": True,
                    "spelling_mismatch": True,
                    "provisional_patient_name": clean_name,
                    "reason": "The phone audio did not verify the exact name after two attempts.",
                    "instruction": (
                        "Stop the spelling loop. Explain that the exact name could not be "
                        "verified by phone audio and offer human-assisted completion."
                    ),
                }

            return {
                "success": False,
                "requires_name_spelling": True,
                "spelling_mismatch": True,
                "provisional_patient_name": clean_name,
                "reason":
                    "The spoken letter sequence did not match the "
                    "patient name supplied to the booking tool.",
                "instruction":
                    "Ask once about only the uncertain name part. "
                    "Keep any correctly captured parts."
            }


        return {
            "success": False,
            "requires_name_spelling": True,
            "spelling_mismatch": False,
            "provisional_patient_name": clean_name,
            "reason":
                "The patient's full-name spelling must be verified "
                "before booking preparation.",
            "instruction":
                "Ask the patient once to spell the name letter by letter. "
                "Store separately spoken name parts and do not restart the process."
        }


    # ========================================================
    # TRANSACTION CONFIRMATION SAFETY GATE
    # ========================================================

    def _normalize_confirmation_text(
        self,
        patient_text
    ):

        normalized = "".join(
            character.lower()
            if (
                character.isalnum()
                or character.isspace()
            )
            else " "
            for character in str(
                patient_text
            )
        )


        return " ".join(
            normalized.split()
        )


    def _is_explicit_rejection(
        self,
        patient_text
    ):

        normalized = (
            self._normalize_confirmation_text(
                patient_text
            )
        )

        if not normalized:

            return False


        negative_exact = {
            "no",
            "no thanks",
            "no thank you",
            "nope",
            "nah",
            "not yet",
            "not now",
            "wait",
            "wait please",
            "stop",
            "stop please",
            "hold on",
            "hold on please",
            "hold off",
            "do not proceed",
            "don t proceed",
            "i do not confirm",
            "i don t confirm",
            "do not book",
            "don t book",
            "do not reschedule",
            "don t reschedule",
            "do not cancel",
            "don t cancel",
            "i changed my mind",
            "i change my mind",
            "never mind",
            "not correct",
            "not confirmed",
            "that is not correct",
            "that s not correct",
            "that is wrong",
            "that s wrong",
        }

        if normalized in negative_exact:

            return True


        negative_starts = (
            "no ",
            "nope ",
            "nah ",
            "wait ",
            "stop ",
            "hold on ",
            "hold off ",
            "do not ",
            "don t ",
            "i do not ",
            "i don t ",
        )

        if normalized.startswith(
            negative_starts
        ):

            return True


        padded = (
            " "
            + normalized
            + " "
        )

        negative_phrases = (
            " do not ",
            " don t ",
            " not correct ",
            " not confirmed ",
            " changed my mind ",
            " change my mind ",
            " never mind ",
        )

        return any(
            phrase in padded
            for phrase in negative_phrases
        )


    def _is_explicit_confirmation(
        self,
        patient_text,
        action
    ):

        normalized = (
            self._normalize_confirmation_text(
                patient_text
            )
        )

        if not normalized:

            return False


        if self._is_explicit_rejection(
            patient_text
        ):

            return False


        padded = (
            " "
            + normalized
            + " "
        )

        unsafe_or_ambiguous = (
            " but ",
            " instead ",
            " actually ",
            " different ",
            " maybe ",
            " perhaps ",
            " not sure ",
            " unsure ",
            " i think ",
            " probably ",
            " possibly ",
            " later ",
            " hold on ",
            " wait ",
            " before ",
            " question ",
            " if ",
        )

        if any(
            phrase in padded
            for phrase in unsafe_or_ambiguous
        ):

            return False


        words = normalized.split()
        word_set = set(words)

        if word_set.intersection(
            {
                "no",
                "not",
                "never",
            }
        ):

            return False


        generic_confirmation = {
            "yes",
            "yes please",
            "yeah",
            "yeah please",
            "yep",
            "yep please",
            "yup",
            "yup please",
            "ok",
            "okay",
            "okay please",
            "alright",
            "all right",
            "sure",
            "sure please",
            "certainly",
            "absolutely",
            "definitely",
            "confirm",
            "confirmed",
            "yes confirmed",
            "i confirm",
            "i confirm it",
            "correct",
            "yes correct",
            "that is correct",
            "that s correct",
            "yes that is correct",
            "yes that s correct",
            "all good",
            "yes all good",
            "sounds good",
            "that sounds good",
            "works for me",
            "that works for me",
            "fine",
            "fine by me",
            "that is fine",
            "that s fine",
            "go ahead",
            "please go ahead",
            "go for it",
            "proceed",
            "do proceed",
            "proceed on",
            "please proceed",
            "yes proceed",
            "yes please proceed",
            "yes please proceed on",
            "do it",
            "please do it",
            "let s do it",
        }

        if normalized in generic_confirmation:

            return True


        positive_words = {
            "yes",
            "yeah",
            "yep",
            "yup",
            "ok",
            "okay",
            "alright",
            "sure",
            "certainly",
            "absolutely",
            "definitely",
            "confirm",
            "confirmed",
            "correct",
            "agreed",
            "fine",
            "proceed",
        }

        polite_reply_words = {
            "yes",
            "yeah",
            "yep",
            "yup",
            "ok",
            "okay",
            "alright",
            "sure",
            "certainly",
            "absolutely",
            "definitely",
            "confirm",
            "confirmed",
            "correct",
            "agreed",
            "fine",
            "please",
            "all",
            "good",
            "sounds",
            "great",
            "that",
            "this",
            "is",
            "s",
            "works",
            "work",
            "for",
            "me",
            "go",
            "ahead",
            "proceed",
            "do",
            "it",
            "on",
            "then",
        }

        if (
            len(words) <= 8
            and word_set.intersection(
                positive_words
            )
            and word_set.issubset(
                polite_reply_words
            )
        ):

            return True


        current_action = str(
            action
            or ""
        ).strip().lower()

        action_words = {
            "book": {
                "book",
                "booking",
                "schedule",
                "scheduling",
            },
            "reschedule": {
                "reschedule",
                "rescheduling",
                "move",
                "moving",
                "change",
                "changing",
            },
            "cancel": {
                "cancel",
                "cancellation",
                "remove",
                "removing",
            },
        }

        current_words = (
            action_words.get(
                current_action,
                set()
            )
        )

        conflicting_words = set()

        for action_name, candidate_words in (
            action_words.items()
        ):

            if action_name != current_action:

                conflicting_words.update(
                    candidate_words
                )

        if word_set.intersection(
            conflicting_words
        ):

            return False


        action_present = bool(
            word_set.intersection(
                current_words
            )
        )

        imperative_starts = {
            "book": (
                "book ",
                "please book ",
                "schedule ",
                "please schedule ",
            ),
            "reschedule": (
                "reschedule ",
                "please reschedule ",
                "move ",
                "please move ",
                "change ",
                "please change ",
            ),
            "cancel": (
                "cancel ",
                "please cancel ",
                "remove ",
                "please remove ",
            ),
        }

        affirmative_present = bool(
            word_set.intersection(
                positive_words
            )
        )

        explicit_imperative = (
            (normalized + " ").startswith(
                imperative_starts.get(
                    current_action,
                    ()
                )
            )
        )

        return (
            action_present
            and (
                affirmative_present
                or explicit_imperative
            )
        )


    def _prepare_before_lookup_repair(
        self,
        action,
        arguments,
        details,
        confirmed
    ):

        candidate = {
            "action":
                action,

            "arguments":
                arguments,

            "details":
                details,
        }


        pending = (
            self.session_state.get(
                "pending_action"
            )
        )


        # ----------------------------------------------------
        # FIRST CALL:
        # PREPARE ONLY - NO DATABASE WRITE
        # ----------------------------------------------------

        if not confirmed:

            self.session_state[
                "pending_action"
            ] = candidate


            return {
                "success": False,

                "requires_confirmation": True,

                "action":
                    action,

                "reason":
                    "Explicit patient confirmation is required "
                    "before this database change.",

                "confirmation_details":
                    details,
            }


        # ----------------------------------------------------
        # CONFIRMATION CANNOT EXECUTE A DIFFERENT ACTION
        # ----------------------------------------------------

        if (
            not isinstance(
                pending,
                dict
            )
            or
            pending.get(
                "action"
            )
            != action
            or
            pending.get(
                "arguments"
            )
            != arguments
        ):

            self.session_state[
                "pending_action"
            ] = candidate


            return {
                "success": False,

                "requires_confirmation": True,

                "action":
                    action,

                "reason":
                    "The transaction details changed or were "
                    "not previously prepared. Ask the patient "
                    "to confirm these exact details.",

                "confirmation_details":
                    details,
            }


        # ----------------------------------------------------
        # SAME PREPARED ACTION + REAL LATER CONFIRMATION
        # DATABASE WRITE IS NOW ALLOWED ONCE
        # ----------------------------------------------------

        self.session_state[
            "pending_action"
        ] = None


        return None



    # ========================================================
    # REAL TOOL: BOOK
    # ========================================================

    def _book_appointment_before_nameflow(
        self,
        patient_name,
        slot_id,
        confirmed=False,
        consultation_mode=None,
        insurance_status=None,
        certificate_purpose=None,
        requested_date_range=None,
        reason_for_visit=None,
    ):

        matching_slot = next(
            (
                slot for slot in self.session_state["last_available_slots"]
                if slot["slot_id"] == slot_id
            ),
            None,
        )
        if not matching_slot:
            return {
                "success": False,
                "reason": "The selected slot was not one of the currently offered slots.",
            }
        if not slot_is_in_future(matching_slot):
            return {
                "success": False,
                "reason": "The selected appointment time has passed. Please choose a current future slot.",
            }

        is_certificate = (
            matching_slot.get("service_name") == "Medical Certificate Consultation"
        )
        certificate_details = None
        if is_certificate:
            mode = str(consultation_mode or "").strip().lower()
            insurance = str(insurance_status or "").strip().lower()
            purpose = str(certificate_purpose or "").strip()
            date_range = str(requested_date_range or "").strip()
            missing = []
            if mode not in {"in_person", "video", "audio"}:
                missing.append("consultation mode")
            if insurance not in {"insured", "not_insured"}:
                missing.append("medical insurance status")
            if not purpose:
                missing.append("certificate purpose")
            if not date_range:
                missing.append("requested certificate date range")
            if missing:
                return {
                    "success": False,
                    "requires_medical_certificate_details": True,
                    "missing_fields": missing,
                    "instruction": "Ask only for the missing medical-certificate information.",
                }
            if mode != matching_slot.get("consultation_mode"):
                return {
                    "success": False,
                    "reason": "The selected slot does not match the requested consultation mode.",
                }
            certificate_details = {
                "consultation_mode": mode,
                "insurance_status": insurance,
                "prototype_fee_aud": 0 if insurance == "insured" else 45,
                "certificate_purpose": purpose,
                "requested_date_range": date_range,
                "certificate_not_guaranteed": True,
            }

        name_spelling_result = self._check_patient_name_spelling(
            patient_name=patient_name,
            slot_id=slot_id,
        )
        if name_spelling_result is not None:
            return name_spelling_result

        booking_arguments = {
            "patient_name": patient_name,
            "slot_id": slot_id,
        }
        confirmation_details = {
            "patient_name": patient_name,
            "provider": matching_slot.get("provider"),
            "service_name": matching_slot.get("service_name"),
            "display_date": matching_slot.get("display_date"),
            "display_time": matching_slot.get("display_time"),
        }
        if reason_for_visit is not None:
            booking_arguments["reason_for_visit"] = reason_for_visit
            confirmation_details["reason_for_visit"] = reason_for_visit
        if certificate_details:
            booking_arguments.update(certificate_details)
            confirmation_details.update(certificate_details)

        confirmation_result = self._prepare_transaction_confirmation(
            action="book",
            arguments=booking_arguments,
            details=confirmation_details,
            confirmed=confirmed,
        )
        if confirmation_result is not None:
            return confirmation_result

        response = supabase.rpc(
            "book_appointment_with_reason_atomic" if reason_for_visit is not None else "book_appointment_atomic",
            dict(p_patient_name=patient_name, p_slot_id=slot_id, p_reason_for_visit=reason_for_visit)
            if reason_for_visit is not None else {"p_patient_name": patient_name, "p_slot_id": slot_id},
        ).execute()
        result = response.data

        if isinstance(result, dict) and result.get("success") is True:
            self.session_state["patient_name"] = patient_name
            self.session_state["selected_slot_id"] = slot_id
            appointment_id = result.get("appointment_id")
            self.session_state["active_appointment_id"] = appointment_id
            if certificate_details and appointment_id is not None:
                try:
                    supabase.table("medical_certificate_requests").upsert(
                        {
                            "appointment_id": appointment_id,
                            "consultation_mode": certificate_details["consultation_mode"],
                            "insurance_status": certificate_details["insurance_status"],
                            "prototype_fee_aud": certificate_details["prototype_fee_aud"],
                            "certificate_purpose": certificate_details["certificate_purpose"],
                            "requested_date_range": certificate_details["requested_date_range"],
                            "prototype_only": True,
                        },
                        on_conflict="appointment_id",
                    ).execute()
                    result["medical_certificate_request_saved"] = True
                    result["prototype_fee_aud"] = certificate_details["prototype_fee_aud"]
                except Exception as error:
                    result["medical_certificate_request_saved"] = False
                    result["intake_warning"] = (
                        "The appointment was booked, but prototype certificate intake "
                        "metadata could not be saved: " + str(error)
                    )
        if isinstance(result, dict) and result.get('success') is True:
            self.session_state['nameflow_booking_arguments'] = None
        if isinstance(result, dict) and result.get('success') is False and result.get('reason') == 'Slot is no longer available':
            return self._recover_taken_slot_v1(result, matching_slot)
        return result


    # ========================================================
    # REAL TOOL: FIND APPOINTMENTS
    # ========================================================

    def _find_before_lookup_repair(
        self,
        patient_name
    ):

        response = (
            supabase
            .table("appointments")
            .select(
                "appointment_id,"
                "patient_name,"
                "doctor_id,"
                "slot_id,"
                "status"
            )
            .ilike(
                "patient_name",
                patient_name
            )
            .eq(
                "status",
                "booked"
            )
            .order(
                "appointment_id"
            )
            .execute()
        )


        appointments = (
            response.data or []
        )

        results = []


        for appointment in appointments:

            slot_response = (
                supabase
                .table("slots")
                .select(
                    "slot_id,"
                    "doctor_id,"
                    "service_id,"
                    "appointment_date,"
                    "appointment_time"
                )
                .eq(
                    "slot_id",
                    appointment["slot_id"]
                )
                .single()
                .execute()
            )

            slot = (
                slot_response.data
            )


            doctor_response = (
                supabase
                .table("doctors")
                .select(
                    "doctor_id,"
                    "doctor_name,"
                    "specialty"
                )
                .eq(
                    "doctor_id",
                    appointment["doctor_id"]
                )
                .single()
                .execute()
            )

            doctor = (
                doctor_response.data
            )


            service_name = None

            if slot:

                service = service_by_id.get(
                    slot["service_id"]
                )

                if service:

                    service_name = (
                        service["service_name"]
                    )


            results.append(
                {
                    "appointment_id":
                        appointment[
                            "appointment_id"
                        ],

                    "patient_name":
                        appointment[
                            "patient_name"
                        ],

                    "service_name":
                        service_name,

                    "provider":
                        doctor["doctor_name"]
                        if doctor
                        else None,

                    "slot_id":
                        appointment[
                            "slot_id"
                        ],

                    "display_date":
                        format_date(
                            slot[
                                "appointment_date"
                            ]
                        )
                        if slot
                        else None,

                    "display_time":
                        format_time(
                            slot[
                                "appointment_time"
                            ]
                        )
                        if slot
                        else None,
                }
            )


        self.session_state[
            "patient_name"
        ] = patient_name

        self.session_state[
            "last_found_appointments"
        ] = results


        if len(results) == 1:

            self.session_state[
                "active_appointment_id"
            ] = results[0][
                "appointment_id"
            ]

            self.session_state[
                "active_service"
            ] = results[0][
                "service_name"
            ]


        return {
            "success": True,
            "patient_name":
                patient_name,
            "booked_appointments":
                results,
        }


    # ========================================================
    # REAL TOOL: RESCHEDULE
    # ========================================================

    def reschedule_appointment(
        self,
        appointment_id,
        new_slot_id,
        confirmed=False
    ):

        appointment = next(
            (
                item
                for item in self.session_state[
                    "last_found_appointments"
                ]
                if item["appointment_id"]
                == appointment_id
            ),
            None,
        )


        if not appointment:

            return {
                "success": False,
                "reason":
                    "Appointment has not been validated."
            }


        new_slot = next(
            (
                slot
                for slot in self.session_state[
                    "last_available_slots"
                ]
                if slot["slot_id"]
                == new_slot_id
            ),
            None,
        )


        if not new_slot:

            return {
                "success": False,
                "reason":
                    "New slot was not one of the "
                    "currently offered slots."
            }


        if not slot_is_in_future(
            new_slot
        ):

            return {
                "success": False,
                "reason":
                    "The selected new appointment time has passed. "
                    "Please choose a current future slot."
            }


        if (
            new_slot["service_name"]
            != appointment["service_name"]
        ):

            return {
                "success": False,
                "reason":
                    "Selected slot belongs to "
                    "a different service."
            }


        confirmation_result = (
            self._prepare_transaction_confirmation(
                action=
                    "reschedule",

                arguments={
                    "appointment_id":
                        appointment_id,

                    "new_slot_id":
                        new_slot_id,
                },

                details={
                    "service_name":
                        appointment.get(
                            "service_name"
                        ),

                    "old_provider":
                        appointment.get(
                            "provider"
                        ),

                    "old_display_date":
                        appointment.get(
                            "display_date"
                        ),

                    "old_display_time":
                        appointment.get(
                            "display_time"
                        ),

                    "new_provider":
                        new_slot.get(
                            "provider"
                        ),

                    "new_display_date":
                        new_slot.get(
                            "display_date"
                        ),

                    "new_display_time":
                        new_slot.get(
                            "display_time"
                        ),
                },

                confirmed=
                    confirmed
            )
        )


        if confirmation_result is not None:

            return confirmation_result


        response = (
            supabase
            .rpc(
                "reschedule_appointment_atomic",
                {
                    "p_appointment_id":
                        appointment_id,

                    "p_new_slot_id":
                        new_slot_id,
                }
            )
            .execute()
        )


        result = response.data


        if (
            isinstance(result, dict)
            and result.get("success") is True
        ):

            result["provider"] = (
                new_slot["provider"]
            )

            result["service_name"] = (
                new_slot["service_name"]
            )

            result["display_date"] = (
                new_slot["display_date"]
            )

            result["display_time"] = (
                new_slot["display_time"]
            )

            self.session_state[
                "active_appointment_id"
            ] = appointment_id

            self.session_state[
                "selected_slot_id"
            ] = new_slot_id


        return result


    # ========================================================
    # REAL TOOL: CANCEL
    # ========================================================

    def cancel_appointment(
        self,
        appointment_id,
        confirmed=False
    ):

        appointment = next(
            (
                item
                for item in self.session_state[
                    "last_found_appointments"
                ]
                if item["appointment_id"]
                == appointment_id
            ),
            None,
        )


        if not appointment:

            return {
                "success": False,
                "reason":
                    "Appointment has not been validated."
            }


        confirmation_result = (
            self._prepare_transaction_confirmation(
                action=
                    "cancel",

                arguments={
                    "appointment_id":
                        appointment_id,
                },

                details={
                    "provider":
                        appointment.get(
                            "provider"
                        ),

                    "service_name":
                        appointment.get(
                            "service_name"
                        ),

                    "display_date":
                        appointment.get(
                            "display_date"
                        ),

                    "display_time":
                        appointment.get(
                            "display_time"
                        ),
                },

                confirmed=
                    confirmed
            )
        )


        if confirmation_result is not None:

            return confirmation_result


        response = (
            supabase
            .rpc(
                "cancel_appointment_atomic",
                {
                    "p_appointment_id":
                        appointment_id
                }
            )
            .execute()
        )


        result = response.data


        if (
            isinstance(result, dict)
            and result.get("success") is True
        ):

            result["provider"] = (
                appointment.get(
                    "provider"
                )
            )

            result["service_name"] = (
                appointment.get(
                    "service_name"
                )
            )

            result["display_date"] = (
                appointment.get(
                    "display_date"
                )
            )

            result["display_time"] = (
                appointment.get(
                    "display_time"
                )
            )

            self.session_state[
                "active_appointment_id"
            ] = None

            self.session_state[
                "selected_slot_id"
            ] = None


        return result


    # ========================================================
    # REAL TOOL: ADMIN / RAG
    # ========================================================

    def get_admin_information(
        self,
        question
    ):

        question_text = str(
            question or ""
        ).strip()

        query = question_text.lower()

        topic = "general"
        relevant_categories = set()
        answer_terms = ()
        recommended_team = (
            "clinic administration staff"
        )

        query_words_text = "".join(
            character
            if character.isalnum()
            else " "
            for character in query
        )

        padded_query = (
            " "
            + " ".join(
                query_words_text.split()
            )
            + " "
        )

        if (
            " fee " in padded_query
            or " fees " in padded_query
            or any(
                term in query
                for term in (
                    "cost",
                    "charge",
                    "billing",
                    "payment",
                    "insurance",
                    "medicare",
                )
            )
        ):
            topic = "fees_and_billing"
            relevant_categories = {
                "fees",
                "billing",
            }
            answer_terms = (
                "fee",
                "fees",
                "cost",
                "charge",
                "billing",
                "payment",
                "insurance",
                "medicare",
            )
            recommended_team = (
                "billing or administration staff"
            )

        elif any(
            term in query
            for term in (
                "opening hour",
                "open today",
                "close today",
                "what time",
                "when are you open",
            )
        ):
            topic = "opening_hours"
            relevant_categories = {'opening hours', 'hours'}
            answer_terms = (
                "open",
                "hours",
                "closed",
            )

        elif any(
            term in query
            for term in (
                "location",
                "located",
                "address",
                "where is",
                "directions",
            )
        ):
            topic = "location"
            relevant_categories = {
                "location",
            }
            answer_terms = (
                "location",
                "located",
                "address",
            )

        elif any(
            term in query
            for term in (
                "companion",
                "family member",
                "bring my mother",
                "bring my father",
                "bring someone",
                "support person",
            )
        ):
            topic = "companion_policy"
            relevant_categories = {
                "companions",
                "visitors",
            }
            answer_terms = (
                "companion",
                "family member",
                "support person",
                "visitor",
            )
            recommended_team = (
                "the treating clinical team or administration staff"
            )

        elif any(
            term in query
            for term in (
                "prescription",
                "medication",
                "test result",
                "referral",
                "medical result",
            )
        ):
            topic = "clinical_follow_up"
            relevant_categories = {
                "clinical",
                "prescriptions",
                "referrals",
                "results",
            }
            answer_terms = (
                "prescription",
                "medication",
                "result",
                "referral",
            )
            recommended_team = (
                "the treating clinician or qualified clinical team"
            )

        elif any(
            term in query
            for term in (
                "medical record",
                "health record",
                "privacy",
                "personal information",
                "release form",
            )
        ):
            topic = "records_and_privacy"
            relevant_categories = {
                "records",
                "privacy",
            }
            answer_terms = (
                "record",
                "privacy",
                "personal information",
            )
            recommended_team = (
                "records, privacy or administration staff"
            )

        elif any(
            term in query
            for term in (
                "complaint",
                "complain",
                "feedback",
                "manager",
            )
        ):
            topic = "complaints"
            relevant_categories = {
                "complaints",
            }
            answer_terms = (
                "complaint",
                "feedback",
                "manager",
            )
            recommended_team = (
                "the practice manager"
            )

        elif any(
            term in query
            for term in (
                "appointment",
                "book",
                "booking",
                "cancel",
                "cancellation",
                "reschedule",
                "availability",
            )
        ):
            topic = "appointments"
            relevant_categories = {
                "appointments",
            }
            answer_terms = (
                "appointment",
                "booking",
                "cancel",
                "cancellation",
                "reschedule",
                "rescheduling",
                "availability",
            )
            recommended_team = (
                "scheduling or administration staff"
            )

        elif any(
            term in query
            for term in (
                "service",
                "specialist",
                "doctor",
                "provider",
                "clinic provide",
            )
        ):
            topic = "services"
            relevant_categories = {
                "services",
            }
            answer_terms = (
                "service",
                "services",
                "provides",
                "specialist",
            )

        elif any(
            term in query
            for term in (
                "bring",
                "document",
                "identification",
                "id card",
            )
        ):
            topic = "documents"
            relevant_categories = {
                "documents",
            }
            answer_terms = (
                "bring",
                "document",
                "identification",
            )

        elif any(
            term in query
            for term in (
                "facility",
                "facilities",
                "amenity",
                "amenities",
                "waiting area",
                "waiting room",
                "waiting space",
                "breastfeeding",
                "breast feeding",
                "nursing room",
                "feeding area",
                "mother's room",
                "mothers room",
                "pharmacy",
                "chemist",
                "cafeteria",
                "cafe",
                "food",
                "prayer",
                "multi faith",
                "multi-faith",
                "quiet room",
                "playground",
                "play area",
                "children's area",
                "childrens area",
            )
        ):
            topic = "facilities"
            relevant_categories = {
                "facilities",
            }
            answer_terms = (
                "facility",
                "available",
                "waiting",
                "breastfeeding",
                "breast feeding",
                "pharmacy",
                "chemist",
                "24 hours",
                "7 days",
                "cafeteria",
                "cafe",
                "prayer",
                "multi faith",
                "multi-faith",
                "quiet room",
                "playground",
                "play area",
            )
            recommended_team = (
                "clinic administration staff"
            )

        elif any(
            term in query
            for term in (
                "symptom",
                "pain",
                "dizzy",
                "dizziness",
                "medical advice",
                "diagnosis",
                "treatment",
            )
        ):
            topic = "clinical_guidance"
            relevant_categories = {
                "clinical",
            }
            answer_terms = (
                "symptom",
                "pain",
                "dizzy",
                "diagnosis",
                "treatment",
            )
            recommended_team = (
                "a GP or qualified clinical team"
            )

        relevant_rows = [
            row
            for row in knowledge_rows
            if str(
                row.get(
                    "category",
                    ""
                )
            ).strip().lower()
            in relevant_categories
        ]

        relevant_text = " ".join(
            " ".join(
                [
                    str(row.get("title", "")),
                    str(row.get("content", "")),
                ]
            ).lower()
            for row in relevant_rows
        )

        answer_available = bool(
            relevant_rows
            and any(
                term in relevant_text
                for term in answer_terms
            )
        )

        contact_rows = [
            row
            for row in knowledge_rows
            if str(
                row.get(
                    "category",
                    ""
                )
            ).strip().lower()
            == "contact"
        ]

        contact_text = " ".join(
            str(
                row.get(
                    "content",
                    ""
                )
            ).lower()
            for row in contact_rows
        )

        if (
            "phone" in contact_text
            and "opening hours" in contact_text
        ):
            verified_contact_route = (
                "phone during clinic opening hours"
            )
            recommended_next_step = (
                "Ask clinic staff by phone during clinic "
                "opening hours to connect you with "
                + recommended_team
                + "."
            )
        elif "phone" in contact_text:
            verified_contact_route = "clinic phone"
            recommended_next_step = (
                "Ask clinic staff by phone to connect you with "
                + recommended_team
                + "."
            )
        else:
            verified_contact_route = None
            recommended_next_step = (
                "A member of "
                + recommended_team
                + " needs to confirm this, but no verified "
                "contact channel is stored in the clinic records."
            )

        if answer_available:
            response_instruction = (
                "Answer directly using only the relevant verified "
                "clinic knowledge. Do not add a routine appointment "
                "offer."
            )
        elif topic == "fees_and_billing":
            response_instruction = (
                "Say that the current fee is not listed in the clinic "
                "records. Direct the patient to billing or administration "
                "staff using the verified next step. Do not say 'contact "
                "reception' and do not offer an appointment."
            )
        else:
            response_instruction = (
                "Say briefly that the requested information is not in "
                "the clinic records. Name the recommended team and use "
                "the verified next step. Do not say only 'contact "
                "reception' and do not offer an appointment."
            )

        return {
            "success": True,
            "patient_question":
                question_text,
            "retrieval_mode":
                "query_focused_category_filter",
            "routing_topic":
                topic,
            "answer_available_in_records":
                answer_available,
            "relevant_verified_clinic_knowledge":
                relevant_rows,
            "recommended_team":
                recommended_team,
            "verified_contact_route":
                verified_contact_route,
            "recommended_next_step":
                recommended_next_step,
            "response_instruction":
                response_instruction,
        }


    # ========================================================
    # PROCESS ONE PATIENT TURN
    # ========================================================

    @staticmethod
    def _verified_static_clinic_answer(patient_text):
        normalized = " ".join("".join(c.lower() if c.isalnum() else " " for c in str(patient_text or "")).split())
        questions = {"what are the clinic opening hours", "what are your opening hours", "what are the opening hours", "what are your hours", "clinic opening hours", "opening hours", "when are you open", "what time do you open", "what time does the clinic open", "what time do you close", "what time does the clinic close", "what are the clinic hours"}
        if normalized not in questions:
            faq_entries = [('location', 'clinic location', ('where is the clinic', 'where is the clinic located', 'what is the clinic address', 'what is your address')), ('contact', 'clinic contact information', ('how can i contact the clinic', 'what is the clinic phone number', 'what is your phone number')), ('documents', 'what to bring', ('what should i bring to my appointment', 'what documents should i bring', 'what do i need to bring to my appointment')), ('facilities', 'public cafeteria', ('is there a cafeteria', 'does the clinic have a cafeteria', 'what are the cafeteria opening hours')), ('facilities', 'prayer area', ('is there a prayer area', 'is there a prayer room', 'does the clinic have a prayer area')), ('facilities', 'patient waiting area', ('is there a waiting area', 'does the clinic have a waiting area')), ('facilities', 'private breastfeeding area', ('is there a breastfeeding area', 'is there a private breastfeeding area', 'does the clinic have a breastfeeding area')), ('facilities', 'onsite pharmacy', ('is there an onsite pharmacy', 'does the clinic have a pharmacy', 'what are the pharmacy opening hours'))]
            selected = next(((category, title) for category, title, queries in faq_entries if normalized in queries), None)
            if selected is None:
                return None
            contents = []
            for row in knowledge_rows:
                category = str(row.get("category", "")).strip().lower()
                title = " ".join(str(row.get("title", "")).strip().lower().split())
                content = str(row.get("content") or "").strip()
                if (category, title) == selected and content:
                    contents.append(content)
            unique = list(dict.fromkeys(contents))
            if len(unique) != 1 or len(unique[0]) > 600:
                return None
            return unique[0]
        titles = {"clinic opening hours", "opening hours", "clinic hours", "clinic operating hours"}
        contents = []
        for row in knowledge_rows:
            category = str(row.get("category", "")).strip().lower()
            title = " ".join(str(row.get("title", "")).strip().lower().split())
            content = str(row.get("content") or "").strip()
            if category in {"hours", "opening hours"} and title in titles and content:
                contents.append(content)
        unique = list(dict.fromkeys(contents))
        if len(unique) != 1 or len(unique[0]) > 600:
            return None
        return unique[0]

    def _process_turn_before_nameflow(
        self,
        patient_text,
        speaker_assessment=None,
    ):

        patient_text = (
            str(patient_text).strip()
        )

        if not patient_text:

            return (
                "I'm sorry, I didn't catch that."
            )


        self.current_patient_text = patient_text

        returning_candidate = None
        if isinstance(speaker_assessment, dict):
            returning_candidate = speaker_assessment.get(
                "returning_voice_candidate"
            )
        if (
            returning_candidate
            and not self.session_state.get("returning_voice_candidate_pending")
        ):
            self.session_state["returning_voice_candidate_pending"] = {
                "candidate_patient_name": returning_candidate,
                "trigger_text": patient_text,
                "attempts": 0,
            }
            ai_text = (
                "I may have found a returning-patient voice profile. For privacy, "
                "please say your complete full name. I will not use the profile "
                "unless the name also matches."
            )
            self.messages.append({"role": "user", "content": patient_text})
            self.messages.append({"role": "assistant", "content": ai_text})
            print("RETURNING VOICE CANDIDATE: secondary name check requested")
            return ai_text

        returning_pending = self.session_state.get(
            "returning_voice_candidate_pending"
        )
        if isinstance(returning_pending, dict):
            candidate_name = returning_pending.get("candidate_patient_name", "")
            candidate_key = self._name_letters(candidate_name)
            spoken_key = self._name_letters(patient_text)
            if candidate_key and candidate_key in spoken_key:
                original_request = returning_pending.get("trigger_text", "")
                self.session_state["returning_voice_candidate_pending"] = None
                self.session_state["patient_name"] = candidate_name
                self.messages.append({
                    "role": "system",
                    "content": (
                        "The caller matched an opt-in returning voice profile and "
                        "also spoke the matching full name. Treat this only as a "
                        "returning-patient candidate, not as authorization for a "
                        "transaction. Continue the original request and retain all "
                        "normal spelling and explicit-confirmation gates."
                    ),
                })
                patient_text = original_request
                self.current_patient_text = patient_text
                print("RETURNING VOICE CANDIDATE: spoken full name matched")
            else:
                returning_pending["attempts"] = int(
                    returning_pending.get("attempts") or 0
                ) + 1
                if returning_pending["attempts"] >= 2:
                    self.session_state["returning_voice_candidate_pending"] = None
                    return (
                        "I could not verify the returning profile, so I will continue "
                        "as a new caller without using any previous patient details. "
                        "How can I help you?"
                    )
                return (
                    "That name did not match the returning profile. Please repeat "
                    "your complete full name once, or say that you want to continue "
                    "as a new caller."
                )

        # A voice match is only a safety signal. It never identifies a person.
        if isinstance(speaker_assessment, dict) and speaker_assessment.get("possible_change"):
            original_name = (
                self.session_state.get("verified_patient_name")
                or self.session_state.get("patient_name")
                or "the original patient"
            )
            self.session_state["speaker_change_pending"] = {
                "original_patient_name": original_name,
                "trigger_text": patient_text,
                "similarity": speaker_assessment.get("similarity"),
                "speaker_id": speaker_assessment.get("speaker_id"),
            }
            ai_text = (
                "To keep the request attached to the right person, "
                f"is this appointment or enquiry for {original_name}, for you, or for someone else?"
            )
            self.messages.append({"role": "user", "content": patient_text})
            self.messages.append({"role": "assistant", "content": ai_text})
            print("SPEAKER-CHANGE SAFETY: confirmation requested")
            return ai_text

        pending_speaker_change = self.session_state.get("speaker_change_pending")
        if isinstance(pending_speaker_change, dict):
            lowered = patient_text.lower()
            representative_terms = (
                "on behalf", "for her", "for him", "for my wife", "for my husband",
                "for my mother", "for my father", "for my friend", "representative",
            )
            own_terms = (
                "for myself", "for me", "my own", "my appointment", "i need",
                "i want", "for my health",
            )
            other_terms = (
                "someone else", "another person", "a different person",
                "my child", "my son", "my daughter", "my brother", "my sister",
            )
            if any(term in lowered for term in own_terms):
                trigger_text = pending_speaker_change.get("trigger_text", "")
                self.session_state = {
                    "patient_name": None,
                    "active_service": None,
                    "last_available_slots": [],
                    "selected_slot_id": None,
                    "last_found_appointments": [],
                    "active_appointment_id": None,
                    "pending_action": None,
                    "pending_name_verification": None,
                    "verified_patient_name": None,
                    "verified_patient_name_key": None,
                    "observed_spelled_name_key": None,
                    "observed_spelled_name_confirmed": False,
                    "observed_spelled_name_parts": [],
                    "name_spelling_mismatch_count": 0,
                    "last_name_evidence_text": None,
                    "speaker_change_pending": None,
                    "returning_voice_candidate_pending": None,
                }
                self.messages = [self.messages[0]]
                self.messages.append({
                    "role": "system",
                    "content": (
                        "A new caller has confirmed this is their own enquiry. "
                        "Start a fresh patient context. Do not use or disclose prior caller details. "
                        "Their initial request was: " + trigger_text
                    ),
                })
                print("SPEAKER-CHANGE SAFETY: new patient context created")
            elif any(term in lowered for term in other_terms):
                trigger_text = pending_speaker_change.get("trigger_text", "")
                self.session_state["speaker_change_pending"] = None
                self.messages.append({
                    "role": "system",
                    "content": (
                        "The current speaker says this request is for another person. "
                        "Do not assume that person's identity or reuse the prior patient's name. "
                        "Ask naturally who the appointment or enquiry is for, then continue their "
                        "original request: " + trigger_text
                    ),
                })
                print("SPEAKER-CHANGE SAFETY: third-party subject clarification requested")
            elif any(term in lowered for term in representative_terms):
                self.session_state["speaker_change_pending"] = None
                self.messages.append({
                    "role": "system",
                    "content": (
                        "The different speaker has said they are calling on behalf of the original patient. "
                        "Continue the original patient context, but do not complete any booking, cancellation, or reschedule "
                        "until appointment details are explicitly confirmed."
                    ),
                })
                print("SPEAKER-CHANGE SAFETY: representative context confirmed")
            else:
                original_name = pending_speaker_change.get("original_patient_name", "the original patient")
                return (
                    f"To keep the details correct, is this request for {original_name}, "
                    "for you, or for someone else?"
                )

        static_answer = self._verified_static_clinic_answer(patient_text)
        if static_answer is not None:
            self.messages.append({"role": "user", "content": patient_text})
            self.messages.append({"role": "assistant", "content": static_answer})
            print("VERIFIED STATIC CLINIC ANSWER")
            return static_answer

        self._capture_proactive_name_spelling(
            patient_text
        )


        # ====================================================
        # REAL PATIENT-TURN CONFIRMATION CHECK
        # ====================================================

        current_turn_confirmed = False


        pending_action = (
            self.session_state.get(
                "pending_action"
            )
        )


        if isinstance(
            pending_action,
            dict
        ):

            if self._is_explicit_rejection(
                patient_text
            ):

                self.session_state[
                    "pending_action"
                ] = None


                print(
                    "TRANSACTION CONFIRMATION:"
                    " patient rejected pending action"
                )


            else:

                current_turn_confirmed = (
                    self._is_explicit_confirmation(
                        patient_text,
                        pending_action.get(
                            "action"
                        )
                    )
                )


                print(
                    "TRANSACTION CONFIRMATION:",
                    current_turn_confirmed,
                    "| Action:",
                    pending_action.get(
                        "action"
                    )
                )


        self.messages.append(
            {
                "role": "user",
                "content": patient_text,
            }
        )


        model_session_state = dict(
            self.session_state
        )

        model_session_state["last_available_slots"] = (
            self.session_state.get("last_available_slots")
            or []
        )[:8]

        model_session_state["last_found_appointments"] = (
            self.session_state.get("last_found_appointments")
            or []
        )[:8]

        self.messages.append(
            {
                "role": "system",

                "content":
                    "CURRENT INTERNAL SESSION STATE:\n"
                    + json.dumps(
                        model_session_state,
                        indent=2
                    )
                    +
                    "\nUse this only for continuity. "
                    "Never expose internal IDs."
            }
        )


        qwen_call_number = 0

        for _ in range(8):

            qwen_call_number += 1
            qwen_call_started = time.perf_counter()

            response = (
                qwen.chat.completions.create(
                    model=QWEN_MODEL,
                    messages=self.messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0,
                    extra_body={'enable_thinking': False},
                )
            )

            qwen_call_latency_ms = round(
                (time.perf_counter() - qwen_call_started)
                * 1000,
                3,
            )

            print(
                "QWEN API CALL LATENCY:",
                {
                    "call_number": qwen_call_number,
                    "model": QWEN_MODEL,
                    "milliseconds": qwen_call_latency_ms,
                },
            )


            assistant_message = (
                response
                .choices[0]
                .message
            )


            if not assistant_message.tool_calls:

                ai_text = (
                    assistant_message.content
                    or
                    "I'm sorry, I couldn't generate a response."
                )


                self.messages.append(
                    {
                        "role": "assistant",
                        "content": ai_text,
                    }
                )


                return ai_text


            self.messages.append(
                assistant_message.model_dump(
                    exclude_none=True
                )
            )


            for tool_call in (
                assistant_message.tool_calls
            ):

                tool_name = (
                    tool_call
                    .function
                    .name
                )


                arguments = json.loads(
                    tool_call
                    .function
                    .arguments
                )


                if (
                    tool_name
                    == "check_service_availability"
                ):

                    tool_result = (
                        self.check_service_availability(
                            arguments["service_name"],
                            consultant_gender=arguments.get("consultant_gender"),
                            consultation_mode=arguments.get("consultation_mode"),
                        )
                    )


                elif (
                    tool_name
                    == "book_appointment"
                ):

                    tool_result = (
                        self.book_appointment(
                            patient_name=
                                arguments[
                                    "patient_name"
                                ],

                            slot_id=int(arguments["slot_id"]),
                            confirmed=current_turn_confirmed,
                            consultation_mode=arguments.get("consultation_mode"),
                            insurance_status=arguments.get("insurance_status"),
                            certificate_purpose=arguments.get("certificate_purpose"),
                            requested_date_range=arguments.get("requested_date_range"),
                            reason_for_visit=arguments.get("reason_for_visit"),
                        )
                    )


                elif (
                    tool_name
                    == "find_booked_appointments"
                ):

                    tool_result = (
                        self.find_booked_appointments(
                            arguments[
                                "patient_name"
                            ]
                        )
                    )


                elif (
                    tool_name
                    == "reschedule_appointment"
                ):

                    tool_result = (
                        self.reschedule_appointment(
                            appointment_id=
                                int(
                                    arguments[
                                        "appointment_id"
                                    ]
                                ),

                            new_slot_id=
                                int(
                                    arguments[
                                        "new_slot_id"
                                    ]
                                ),

                            confirmed=
                                current_turn_confirmed,
                        )
                    )


                elif (
                    tool_name
                    == "cancel_appointment"
                ):

                    tool_result = (
                        self.cancel_appointment(
                            appointment_id=
                                int(
                                    arguments[
                                        "appointment_id"
                                    ]
                                ),

                            confirmed=
                                current_turn_confirmed,
                        )
                    )


                elif (
                    tool_name
                    == "get_admin_information"
                ):

                    tool_result = (
                        self.get_admin_information(
                            arguments[
                                "question"
                            ]
                        )
                    )


                else:

                    tool_result = {
                        "success": False,
                        "reason":
                            "Unsupported tool."
                    }


                print()
                print(
                    "===================================="
                )
                print(
                    "RECEPTIONIST TOOL"
                )
                print(
                    "===================================="
                )
                print(
                    "Tool:",
                    tool_name
                )
                console_tool_result = tool_result

                if (
                    tool_name
                    == "check_service_availability"
                    and isinstance(tool_result, dict)
                ):
                    available_slots = (
                        tool_result.get("available_slots")
                        or []
                    )
                    console_tool_result = {
                        "success": tool_result.get("success"),
                        "service_name": tool_result.get("service_name"),
                        "provider_count": len(
                            tool_result.get("providers") or []
                        ),
                        "available_slot_count": len(available_slots),
                        "first_available_slots": available_slots[:3],
                        "query_limit": MAX_AVAILABILITY_ROWS_PER_QUERY,
                    }

                print(
                    json.dumps(
                        console_tool_result,
                        indent=2
                    )
                )


                if tool_name == "book_appointment" and len(assistant_message.tool_calls) == 1 and isinstance(tool_result, dict) and tool_result.get("requires_confirmation"):
                    answer = self._booking_confirmation_text_v1(tool_result['confirmation_details'])
                    self.messages.append({'role': 'tool', 'tool_call_id': tool_call.id,
                                          'name': tool_name, 'content': json.dumps(tool_result)})
                    self.messages.append({'role': 'assistant', 'content': answer})
                    return answer

                if tool_name == 'book_appointment' and len(assistant_message.tool_calls) == 1 and isinstance(tool_result, dict) and (tool_result.get('requires_fresh_slot_selection') or tool_result.get('success') is True):
                    answer = self._booking_result_text_v1(tool_result)
                    self.messages.append({'role':'tool','tool_call_id':tool_call.id,'name':tool_name,'content':json.dumps(tool_result)})
                    self.messages.append({'role':'assistant','content':answer})
                    return answer

                model_tool_result = tool_result

                if (
                    tool_name
                    == "check_service_availability"
                    and isinstance(tool_result, dict)
                ):
                    model_tool_result = dict(tool_result)
                    all_available_slots = (
                        tool_result.get("available_slots")
                        or []
                    )
                    model_tool_result["available_slots"] = (
                        all_available_slots[:8]
                    )
                    model_tool_result["total_slots_retrieved"] = (
                        len(all_available_slots)
                    )

                self.messages.append(
                    {
                        "role": "tool",

                        "tool_call_id":
                            tool_call.id,

                        "content":
                            json.dumps(
                                model_tool_result
                            )
                    }
                )

                if (
                    tool_name
                    == "check_service_availability"
                ):
                    fast_response = (
                        self._fast_availability_response(
                            tool_result
                        )
                    )

                    if fast_response:
                        self.messages.append(
                            {
                                "role": "assistant",
                                "content": fast_response,
                            }
                        )

                        print(
                            "VERIFIED AVAILABILITY FAST RESPONSE:",
                            "second Qwen call skipped"
                        )

                        return fast_response


        return (
            "I'm sorry, I couldn't complete that request."
        )

    def _nameflow_reply(self, patient_text, answer):
        self.messages.append({'role': 'user', 'content': patient_text})
        self.messages.append({'role': 'assistant', 'content': answer})
        return answer

    def _groups_before_lookup_repair(self, text):
        import re
        tokens = re.findall(r'[A-Za-z]+|[,.;!?]', str(text).lower())
        groups, run = [], []
        for token in tokens + ['END']:
            if len(token) == 1 and token.isalpha():
                run.append(token)
            else:
                if len(run) >= 2:
                    groups.append(''.join(run))
                run = []
        if not groups:
            # Reuse existing NATO/spoken-letter support only with explicit cues.
            if any(cue in str(text).lower() for cue in ('as in', 'alpha', 'bravo', 'double', 'letter', 'spell')):
                letters = self._spoken_spelling_letters(text)
                if len(letters) >= 2:
                    groups = [letters]
        return groups

    def _nameflow_readback(self, flow):
        name = ' '.join(part.capitalize() for part in flow['parts'])
        spelling = ', '.join('-'.join(part.upper()) for part in flow['parts'])
        flow.update(stage='readback', candidate=name)
        self.session_state['pending_name_verification'] = {
            'patient_name': name, 'slot_id': (flow.get('booking') or {}).get('slot_id')}
        return f'I have {name}, spelled {spelling}. Is that the patient’s complete name and is the spelling correct?'

    def _nameflow_prompt(self):
        return 'Please say the patient’s full name and spell it letter by letter.'

    def _check_patient_name_spelling(self, patient_name, slot_id):
        if self._name_letters(patient_name) == self.session_state.get('verified_patient_name_key'):
            return None
        flow = self.session_state.get('nameflow_v1')
        if flow is None:
            flow = {'stage': 'collect', 'parts': [], 'attempts': 0,
                    'expected': str(patient_name or ''),
                    'booking': self.session_state.get('nameflow_booking_arguments')}
            self.session_state['nameflow_v1'] = flow
            self.session_state['pending_action'] = None
        return {'success': False, 'requires_name_spelling': True,
                'instruction': 'Name collection is controlled by the server. Do not invent or re-request a verified name.'}

    def _capture_proactive_name_spelling(self, patient_text):
        if self.session_state.get('nameflow_v1') or self.session_state.get('verified_patient_name_key'):
            return
        self._capture_proactive_name_spelling_before_nameflow(patient_text)

    def _book_before_relationship_v1(self, patient_name, slot_id, confirmed=False,
                         consultation_mode=None, insurance_status=None,
                         certificate_purpose=None, requested_date_range=None, reason_for_visit=None):
        if self.session_state.get('speaker_change_pending'):
            return {'success': False, 'requires_patient_subject_clarification': True,
                    'reason': 'Clarify the patient subject before preparing a booking.'}
        if reason_for_visit is None:
            pending = self.session_state.get('pending_action') or {}
            saved = self.session_state.get('nameflow_booking_arguments') or {}
            candidate = pending.get('arguments') or saved
            if candidate.get('slot_id') == slot_id and (pending.get('action') == 'book' or not pending):
                reason_for_visit = candidate.get('reason_for_visit')
        try:
            reason_for_visit = self._validated_booking_reason_v1(reason_for_visit)
        except ValueError as error:
            self.session_state['pending_action'] = None
            return {'success': False, 'requires_corrected_reason': True, 'reason': str(error)}
        arguments = dict(patient_name=patient_name, slot_id=slot_id,
                         consultation_mode=consultation_mode, insurance_status=insurance_status,
                         certificate_purpose=certificate_purpose, requested_date_range=requested_date_range,
                         reason_for_visit=reason_for_visit)
        verified = self.session_state.get('verified_patient_name')
        if verified and self._name_letters(patient_name) != self._name_letters(verified):
            return {'success': False, 'requires_corrected_patient_name': True,
                    'verified_patient_name': verified,
                    'instruction': 'Use the already verified name exactly. Do not ask for spelling again.'}
        if not verified:
            self.session_state['nameflow_booking_arguments'] = arguments
            if self.session_state.get('nameflow_v1'):
                self.session_state['nameflow_v1']['booking'] = arguments
        return self._book_appointment_before_nameflow(confirmed=confirmed, **arguments)

    def _handle_before_lookup_repair(self, text, flow):
        normalized = self._normalize_confirmation_text(text)
        cancellations = {'stop', 'stop please', 'never mind', 'do not book', 'don t book',
                         'cancel this request', 'i do not want to book', 'i don t want to book'}
        if normalized in cancellations:
            self.session_state['nameflow_v1'] = None
            self.session_state['nameflow_booking_arguments'] = None
            self.session_state['pending_name_verification'] = None
            self.session_state['pending_action'] = None
            return 'I have stopped this booking request. No appointment has been created.'
        if flow['stage'] == 'handoff':
            return 'I could not verify the name from this audio. Please use human-assisted completion; I have not created an appointment.'
        yes = {'yes', 'yeah', 'yep', 'correct', 'ok', 'okay', 'yes ok', 'yes okay',
               'yes correct', 'yes that is correct', 'that is correct', 'yes it is correct',
               'yes the spelling is correct', 'the spelling is correct', 'yes please',
               'yes that s correct', 'that s correct', 'yes it s correct', 'yes it is',
               'yes it s ok', 'yes it is ok', 'yes okay correct', 'yup', 'all good'}
        if flow['stage'] == 'readback' and (normalized in yes or self._is_explicit_confirmation(text, 'name')):
            name = flow['candidate']
            key = self._name_letters(name)
            self.session_state.update(patient_name=name, verified_patient_name=name,
                verified_patient_name_key=key, observed_spelled_name_key=key,
                observed_spelled_name_parts=list(flow['parts']), observed_spelled_name_confirmed=True,
                pending_name_verification=None, nameflow_v1=None, pending_action=None)
            self.messages.append({'role': 'system', 'content':
                f'The server verified the complete name and spelling: {name}. Keep this exact name. '
                'Do not ask for name or spelling again. Name confirmation is not transaction confirmation.'})
            booking = flow.get('booking')
            if booking:
                booking = dict(booking, patient_name=name)
                result = self.book_appointment(confirmed=False, **booking)
                if result.get('requires_confirmation'):
                    details = result['confirmation_details']
                    return 'Thank you. ' + self._booking_confirmation_text_v1(details)
                # Preserve missing certificate details and unavailable-slot errors.
                return 'Your name and spelling are verified. ' + str(result.get('reason') or result.get('instruction') or 'Please provide the remaining appointment details.')
            return None
        groups = self._nameflow_groups(text)
        if flow['stage'] == 'readback' and normalized in {'no', 'no that is wrong', 'not correct', 'that is wrong', 'no incorrect'}:
            flow['stage'] = 'collect'
            flow['attempts'] += 1
            if flow['attempts'] >= 2:
                flow['stage'] = 'handoff'
                return 'I could not verify the exact name. Please use human-assisted completion; no appointment has been created.'
            return 'Please spell only the incorrect part, and tell me whether it is the first or last name.'
        if groups:
            old_parts = flow['parts']
            if len(groups) >= 2:
                flow['parts'] = groups
            elif old_parts and ('last name' in normalized or 'surname' in normalized):
                flow['parts'] = old_parts[:-1] + groups
            elif old_parts and 'first name' in normalized:
                flow['parts'] = groups + old_parts[1:]
            elif flow['stage'] == 'readback':
                # An unlabelled correction is ambiguous; never guess which part changed.
                return 'Is that spelling for your first name, last name, or your complete name?'
            elif old_parts:
                if groups[0] not in old_parts:
                    flow['parts'] = old_parts + groups
            else:
                flow['parts'] = groups
            expected_parts = str(flow.get('expected') or '').split()
            if len(flow['parts']) == 1 and len(expected_parts) >= 2:
                joined = flow['parts'][0]
                keys = [self._name_letters(part) for part in expected_parts]
                if joined == ''.join(keys):
                    flow['parts'] = keys
                elif len(keys) == 2 and joined.startswith(keys[0]) and len(joined) > len(keys[0]) + 1:
                    flow['parts'] = [keys[0], joined[len(keys[0]):]]
                elif joined == keys[0]:
                    return 'I have the first name spelling. Please spell only the remaining name part.'
            # Spelled evidence can correct the earlier spoken candidate, but still needs readback.
            return self._nameflow_readback(flow)
        flow['attempts'] += 1
        if flow['attempts'] >= 2:
            flow['stage'] = 'handoff'
            return 'The audio has not captured the letters clearly. Please use human-assisted completion; no appointment has been created.'
        if flow['stage'] == 'readback':
            return 'Please say yes if the name and spelling I read back are correct, or no to correct them.'
        # A misheard name starting with "no" is not treated as cancellation.
        return 'I did not capture the letters clearly. Please spell the name; you do not need to repeat the appointment details.'

    def _process_before_handover_repair(self, patient_text, speaker_assessment=None):
        import re
        text = str(patient_text or '').strip()
        if not text:
            return "I'm sorry, I didn't catch that."
        flow = self.session_state.get('nameflow_v1')
        subject_change = isinstance(speaker_assessment, dict) and (
            speaker_assessment.get('possible_change') or speaker_assessment.get('returning_voice_candidate'))
        if subject_change:
            self.session_state['nameflow_v1'] = None
            self.session_state['nameflow_booking_arguments'] = None
            self.session_state['pending_name_verification'] = None
            flow = None
        normalized = self._normalize_confirmation_text(text)
        groups = self._nameflow_groups(text)
        positive = self._name_spelling_positive(text) and not re.search(r'\b(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|appointment|availability|slot|cancel|reschedule|instead|change)\b|[0-9]', normalized)
        name_rejections = {'no', 'no that is wrong', 'not correct', 'that is wrong', 'no incorrect'}
        stop_requests = {'stop', 'stop please', 'never mind', 'do not book', 'don t book',
                         'cancel this request', 'i do not want to book', 'i don t want to book'}
        # Handle spelling/confirmation directly. All other turns remain normal conversation.
        # Do not classify arbitrary utterances as failed spelling attempts.
        name_turn = bool(groups or positive or normalized in name_rejections | stop_requests
            or re.search(r'\b(?:my name|first name|last name|full name|surname|spell|spelling|letter)\b', normalized))
        delegated_from_flow = False
        if flow and not self.session_state.get('speaker_change_pending'):
            if name_turn:
                self.current_patient_text = text
                result = self._nameflow_handle(text, flow)
                if result is not None:
                    return self._nameflow_reply(text, result)
                text = 'My name and spelling have been verified. Please continue preparing my appointment.'
            else:
                delegated_from_flow = True
                # A changed schedule needs a new exact transaction preparation.
                if re.search(r'\b(?:appointment|available|availability|slot|book|booking|reschedule|today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', normalized):
                    flow['booking'] = None
                    self.session_state['nameflow_booking_arguments'] = None
                    self.session_state['pending_action'] = None
                self.messages.append({'role': 'system', 'content':
                    'The caller is continuing the clinic or scheduling conversation. Answer this exact '
                    'request normally using clinic tools. Do not treat it as failed spelling, and do not '
                    'demand a name for availability or general enquiries. Unfinished name evidence remains '
                    'on hold. If the requested date/time changes, use fresh availability and transaction details.'})
        answer = self._process_turn_before_nameflow(text, speaker_assessment=speaker_assessment)
        flow = self.session_state.get('nameflow_v1')
        asks_name = bool(re.search(r'(?:\bspell (?:your |the |their )?(?:full |complete )?name\b|\b(?:full|complete|first|last) name\b)', answer, re.I)
                         and re.search(r'(?:\?|please|could you|may i|can you)', answer, re.I))
        listing = bool(re.search(r'\b(?:available slots|available appointments|appointments available)\b', answer, re.I)
                       or re.search(r'(?m)^\s*[-*] ', answer))
        if re.search(r'\b(?:book|booking|schedule|reserve)\b', normalized):
            self.session_state['nameflow_booking_requested'] = True
        booking_context = bool(self.session_state.get('last_available_slots') and not listing and
            (self.session_state.get('nameflow_booking_requested') or
             re.search(r'\b(?:book|booking|appointment)\b', answer, re.I)))
        if listing and asks_name:
            # Preserve real offered slots, remove premature identity collection.
            response = re.sub(r'[^.!?\n]*\b(?:full|complete) name\b[^.!?\n]*[.!?]?', '', answer, flags=re.I).strip()
            response += '\n\nWhich offered appointment would you prefer?'
            if self.messages and self.messages[-1].get('role') == 'assistant':
                self.messages[-1]['content'] = response
            return response
        if (flow and not delegated_from_flow) or (asks_name and booking_context and not self.session_state.get('verified_patient_name_key')):
            if flow is None:
                flow = {'stage': 'collect', 'parts': [], 'attempts': 0,
                        'expected': self.session_state.get('patient_name') or '',
                        'booking': self.session_state.get('nameflow_booking_arguments')}
                self.session_state['nameflow_v1'] = flow
                self.session_state['pending_action'] = None
            response = self._nameflow_prompt()
            if groups:
                flow['parts'] = groups
                response = self._nameflow_readback(flow)
            if self.messages and self.messages[-1].get('role') == 'assistant':
                self.messages[-1]['content'] = response
            return response
        if asks_name and self.session_state.get('verified_patient_name_key'):
            response = 'Your name and spelling are already verified. Please tell me which offered appointment you would like to book.'
            if self.messages and self.messages[-1].get('role') == 'assistant':
                self.messages[-1]['content'] = response
            return response
        return answer

    def _process_before_lookup_repair(self, patient_text, speaker_assessment=None):
        import re
        text = str(patient_text or '').strip()
        pending = self.session_state.get('speaker_change_pending')
        fresh_change = isinstance(speaker_assessment, dict) and speaker_assessment.get('possible_change')
        if not isinstance(pending, dict) or fresh_change:
            return self._process_before_handover_repair(text, speaker_assessment)
        normalized = self._normalize_confirmation_text(text)
        representative = bool(re.search(
            r'\b(?:on behalf|for her|for him|for them|for my wife|for my husband|for my mother|for my father|for my friend|same patient|original patient|representative)\b', normalized))
        own = bool(re.search(r'\b(?:for myself|for me|my own|my appointment|for my health)\b', normalized))
        other = bool(re.search(r'\b(?:someone else|another person|different person|my child|my son|my daughter|my brother|my sister)\b', normalized))
        # Do not use generic "I want" / "I need" to override "on behalf".
        # Reject explicit denials or competing patient-subject statements.
        denied = bool(re.search(r'\b(?:not|don t|do not)(?: \w+){0,3}\s+(?:on behalf|for her|for him|for them|for myself|for me|same patient|the original patient)\b', normalized))
        if denied or sum((representative, own, other)) != 1:
            answer = 'Is this enquiry for the original patient, for you, or for someone else? Please state which patient the request is for.'
        elif representative:
            self.session_state['speaker_change_pending'] = None
            self.current_patient_text = text
            self.messages.append({'role': 'system', 'content':
                'The caller states they are continuing for the original patient. This does not establish identity. '
                'Retain the patient context. Name verification and a separate explicit transaction confirmation remain required.'})
            answer = ('Thank you. We can continue the original patient\'s request. '
                      'Please confirm the appointment details I read back if you want to proceed.'
                      if self.session_state.get('pending_action') else
                      'Thank you. We can continue the original patient\'s enquiry. How can I help?')
        else:
            # New subject: discard ALL prior patient, spelling and transaction
            # state, including v3 nameflow fields. Keep only system instructions.
            self.session_state = {
                'patient_name': None, 'active_service': None, 'last_available_slots': [],
                'selected_slot_id': None, 'last_found_appointments': [],
                'active_appointment_id': None, 'pending_action': None,
                'pending_name_verification': None, 'verified_patient_name': None,
                'verified_patient_name_key': None, 'observed_spelled_name_key': None,
                'observed_spelled_name_confirmed': False, 'observed_spelled_name_parts': [],
                'name_spelling_mismatch_count': 0, 'last_name_evidence_text': None,
                'speaker_change_pending': None, 'returning_voice_candidate_pending': None,
            }
            self.current_patient_text = text
            self.messages = [self.messages[0]]
            self.messages.append({'role': 'system', 'content':
                'The caller clarified a separate patient subject. Previous patient and pending transaction context was cleared. '
                'Collect the new request without treating voice similarity as identity.'})
            answer = ('We will start a separate request for you. How can I help?' if own else
                      'We will start a separate request for that person. How can I help?')
        self.messages.append({'role': 'user', 'content': text})
        self.messages.append({'role': 'assistant', 'content': answer})
        return answer

    def _nameflow_groups(self, text):
        import re
        spoken = {'ay':'a','bee':'b','cee':'c','see':'c','dee':'d','ee':'e','ef':'f','gee':'g',
                  'aitch':'h','eye':'i','jay':'j','kay':'k','el':'l','em':'m','en':'n','oh':'o',
                  'pee':'p','cue':'q','queue':'q','are':'r','ess':'s','tee':'t','you':'u',
                  'vee':'v','ex':'x','why':'y','zee':'z','zed':'z'}
        raw = str(text).lower().replace('’', "'")
        raw = re.sub(r"(?<=\w)'(?=\w)", '', raw)
        raw = re.sub(r'\bi\s+i\s+(?=need\b|want\b|would\b|am\b|have\b|feel\b|was\b)', 'i ', raw)
        tokens = re.findall(r'[A-Za-z]+|[,.;!?]', raw)
        groups, run = [], []
        literal = 0
        for token in tokens + ['END']:
            if len(token) == 1 and token.isalpha():
                run.append(token); literal += 1
            elif token in spoken:
                run.append(spoken[token])
            else:
                # Word-letter names are accepted beside literal letter evidence,
                # not extracted from ordinary phrases such as "you are".
                if len(run) >= 2 and literal >= 2:
                    groups.append(''.join(run))
                run, literal = [], 0
        if not groups:
            return self._groups_before_lookup_repair(raw)
        return groups

    def _name_spelling_positive(self, text):
        import re
        normalized = self._normalize_confirmation_text(text)
        if self._is_explicit_rejection(text) or re.search(r'\b(?:but|wrong|incorrect|not|no|unsure)\b', normalized):
            return False
        return (self._is_explicit_confirmation(text, 'name') or
                bool(re.match(r'^(?:yes|yeah|yep|yup|okay|ok|correct|all good)(?:$| )', normalized)))

    def _nameflow_handle(self, text, flow):
        groups = self._nameflow_groups(text)
        if groups and flow.get('stage') == 'readback':
            # "Yes, T A ..." is fresh spelling evidence, never assent to a
            # different candidate. Preserve first/last correction labels.
            normalized = self._normalize_confirmation_text(text)
            if len(groups) == 1 and len(flow.get('parts') or []) == 1 and ('last name' in normalized or 'surname' in normalized):
                flow['parts'] += groups
                return self._nameflow_readback(flow)
            label = ('last name ' if 'last name' in normalized or 'surname' in normalized else
                     'first name ' if 'first name' in normalized else '')
            text = label + ', '.join(' '.join(part) for part in groups)
        elif flow.get('stage') == 'readback' and self._name_spelling_positive(text):
            text = 'yes'
        return self._handle_before_lookup_repair(text, flow)

    def _lookup_reply(self, text, answer):
        self.current_patient_text = text
        return self._nameflow_reply(text, answer)

    def _lookup_readback(self, flow):
        name = ' '.join(p.capitalize() for p in flow['parts'])
        flow.update(stage='readback', candidate=name)
        letters = ', '.join('-'.join(p.upper()) for p in flow['parts'])
        return f'I have {name}, spelled {letters}. Is that the patient’s complete name and is the spelling correct?'

    def _lookup_verified_results(self, name):
        # Use the confirmed spelling deterministically, even when an LLM supplies
        # a different phonetic guess. This function performs read-only lookup.
        result = self._find_before_lookup_repair(name)
        self.session_state['pending_action'] = None
        self.session_state['lookup_verified_name'] = name
        records = result.get('booked_appointments') or []
        self.messages.append({'role':'system','content':
            'Read-only appointment lookup used the server-confirmed name exactly: '+name+'. Result: '+
            __import__('json').dumps(result)+'. Name confirmation is not transaction confirmation.'})
        if not records:
            self.session_state['active_appointment_id'] = None
            return (f'I found no exact booked-appointment match for {name}. '
                    'The stored booking may use a different spelling. Please spell the full name exactly as it appears on your booking. '
                    'I have not changed any appointment.')
        lines = []
        for item in records[:5]:
            lines.append(f"Appointment {item.get('appointment_id')}: {item.get('service_name')} with {item.get('provider')} on {item.get('display_date')} at {item.get('display_time')}.")
        action = self.session_state.get('lookup_operation')
        question = ('Which appointment would you like to cancel?' if action == 'cancel' else
                    'Which appointment would you like to reschedule, and what new date or time would you prefer?' if action == 'reschedule' else
                    'Which appointment would you like help with?')
        return '\n'.join(lines) + '\n' + question

    def _find_before_relationship_v1(self, patient_name):
        verified = self.session_state.get('verified_patient_name')
        if not verified or self.session_state.get('lookup_nameflow'):
            if not self.session_state.get('lookup_nameflow'):
                self.session_state['lookup_nameflow'] = {'stage':'collect','parts':[], 'attempts':0}
            return {'success':False, 'requires_name_spelling':True,
                    'instruction':'Collect spelling and read it back through the server name flow before lookup. Do not invent a spelling.'}
        result = self._find_before_lookup_repair(verified)
        self.session_state['lookup_verified_name'] = verified
        if not result.get('booked_appointments'):
            self.session_state['active_appointment_id'] = None
            result.update(no_exact_name_match=True,
                instruction='There is no exact match for the confirmed spelling. Ask for the name exactly as recorded on the booking; never invent a similar name or select another patient automatically.')
        return result

    def _prepare_before_relationship_v1(self, action, arguments, details, confirmed):
        if confirmed and self.session_state.get('lookup_nameflow'):
            return {'success':False, 'requires_name_spelling':True,
                    'reason':'Name spelling is awaiting confirmation. No appointment change has been authorised.'}
        return self._prepare_before_lookup_repair(action, arguments, details, confirmed)

    def _process_before_relationship_v1(self, patient_text, speaker_assessment=None):
        import re
        text = str(patient_text or '').strip()
        normalized = self._normalize_confirmation_text(text)
        if (isinstance(speaker_assessment, dict) and
                (speaker_assessment.get('possible_change') or speaker_assessment.get('returning_voice_candidate'))):
            self.session_state['lookup_nameflow'] = None
            self.session_state['lookup_verified_name'] = None
            self.session_state['lookup_operation'] = None
            return self._process_before_lookup_repair(text, speaker_assessment)
        if self.session_state.get('speaker_change_pending'):
            return self._process_before_lookup_repair(text, speaker_assessment)
        flow = self.session_state.get('lookup_nameflow')
        intent = ('reschedule' if re.search(r'\b(?:reschedule|rescheduling|change my appointment|move my appointment)\b', normalized) else
                  'cancel' if re.search(r'\b(?:cancel|cancellation)\b.*\b(?:appointment|booking)\b', normalized) else
                  'lookup' if re.search(r'\b(?:find|look up|check my)\b.*\b(?:appointment|booking)\b', normalized) else None)
        if intent:
            self.session_state['lookup_operation'] = intent
        groups = self._nameflow_groups(text)
        correction = bool(groups and self.session_state.get('lookup_operation') and not self.session_state.get('nameflow_v1'))
        if (intent and not self.session_state.get('verified_patient_name')) or correction:
            if flow is None:
                flow = {'stage':'collect','parts':[], 'attempts':0}
                self.session_state['lookup_nameflow'] = flow
        if flow:
            stop = normalized in {'stop','stop please','never mind','cancel this request','do not proceed','don t proceed'}
            if stop:
                self.session_state['lookup_nameflow'] = None
                self.session_state['lookup_operation'] = None
                self.session_state['pending_action'] = None
                return self._lookup_reply(text, 'I have stopped this request. No appointment has been changed.')
            if groups:
                self.session_state['pending_action'] = None
                self.session_state['last_found_appointments'] = []
                self.session_state['active_appointment_id'] = None
                if len(groups) >= 2:
                    flow['parts'] = groups
                elif flow['parts'] and ('last name' in normalized or 'surname' in normalized):
                    flow['parts'] = (flow['parts'][:-1] if len(flow['parts']) >= 2 else flow['parts']) + groups
                elif flow['parts'] and 'first name' in normalized:
                    flow['parts'] = groups + flow['parts'][1:]
                elif flow['parts'] and flow['stage'] == 'readback':
                    return self._lookup_reply(text, 'Is that spelling for your first name, last name, or complete name?')
                elif flow['parts']:
                    flow['parts'] += groups
                else:
                    flow['parts'] = groups
                flow['attempts'] = 0
                return self._lookup_reply(text, self._lookup_readback(flow))
            rejected = self._is_explicit_rejection(text)
            if flow['stage'] == 'readback' and rejected:
                flow['stage'] = 'collect'
                return self._lookup_reply(text, 'Please spell the incorrect part and say whether it is your first or last name.')
            if flow['stage'] == 'readback' and self._name_spelling_positive(text):
                name = flow['candidate'];key = self._name_letters(name)
                self.session_state.update(patient_name=name, verified_patient_name=name,
                    verified_patient_name_key=key, observed_spelled_name_key=key,
                    observed_spelled_name_parts=list(flow['parts']), observed_spelled_name_confirmed=True,
                    pending_name_verification=None, pending_action=None, lookup_nameflow=None)
                answer = self._lookup_verified_results(name)
                return self._lookup_reply(text, answer)
            # Scheduling/general questions must remain answerable during capture.
            if re.search(r'\b(?:available|availability|today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|opening|hours|services)\b', normalized) and not intent:
                return self._process_before_lookup_repair(text, speaker_assessment)
            if not intent:
                flow['attempts'] += 1
            if flow['attempts'] >= 2:
                return self._lookup_reply(text, 'I could not capture the spelling reliably. Please use human-assisted completion; no appointment has been changed.')
            return self._lookup_reply(text, 'To find your existing booking, please spell your complete full name letter by letter, with a pause between the first and last name.')
        answer = self._process_before_lookup_repair(text, speaker_assessment)
        if self.session_state.get('lookup_operation') and answer.startswith('Your name and spelling are already verified'):
            action = self.session_state['lookup_operation']
            answer = ('Your name and spelling are already confirmed. Which new date or time would you prefer?' if action == 'reschedule' else
                      'Your name and spelling are already confirmed. Which appointment would you like to cancel?' if action == 'cancel' else
                      'Your name and spelling are already confirmed. Which appointment would you like help with?')
            if self.messages and self.messages[-1].get('role') == 'assistant':
                self.messages[-1]['content'] = answer
        if self.session_state.get('lookup_nameflow'):
            answer = 'To find your existing booking, please spell your complete full name letter by letter, with a pause between the first and last name.'
            if self.messages and self.messages[-1].get('role') == 'assistant':
                self.messages[-1]['content'] = answer
        return answer

    def _validated_booking_reason_v1(self, candidate):
        if candidate is None or not str(candidate).strip():
            return None
        quote = ' '.join(str(candidate).split())
        if len(quote) > 1000:
            raise ValueError('Please use a shorter verbatim excerpt of the caller-stated reason.')
        # Search only patient utterances in the current patient context.
        # Subject-change handling clears messages; system/model claims are excluded.
        utterances = [str(m.get('content') or '') for m in self.messages if m.get('role') == 'user']
        current = getattr(self, 'current_patient_text', '')
        if not self.session_state.get('speaker_change_pending'):
            utterances.append(str(current or ''))
        for utterance in utterances:
            if quote.casefold() in ' '.join(utterance.split()).casefold():
                return quote
        raise ValueError('The reason must quote the patient exactly. Do not infer a diagnosis or use assistant wording.')

    def _booking_confirmation_text_v1(self, details):
        text = (f"I have prepared the booking for {details.get('patient_name')} with "
                f"{details.get('provider')} for {details.get('service_name')} on "
                f"{details.get('display_date')} at {details.get('display_time')}. ")
        if details.get('reason_for_visit'):
            text += f"The reason you gave is: {details['reason_for_visit']}. "
        if 'certificate_purpose' in details:
            text += (f"Consultation mode: {details.get('consultation_mode')}. "
                     f"Insurance: {details.get('insurance_status')}. "
                     f"Certificate purpose: {details.get('certificate_purpose')}. "
                     f"Requested date range: {details.get('requested_date_range')}. "
                     f"Prototype fee: {details.get('prototype_fee_aud')} Australian dollars. "
                     'A certificate is not guaranteed. ')
        return text + 'Do you confirm this booking?'

    def _handover_subject_v1(self, text):
        import re
        value = self._normalize_confirmation_text(text)
        original = bool(re.search(r'\b(?:on behalf|original patient|same patient|for the patient|for her|for him|for them|(?:her|his|their) (?:appointment|booking))\b', value))
        own = bool(re.search(r'\b(?:for myself|for me|my own|my appointment|under my name|in my name)\b', value))
        other = bool(re.search(r'\b(?:someone else|another patient|another person|different patient)\b', value))
        denied = bool(re.search(r'\b(?:not|don t|do not)\b', value))
        if denied or sum((original,own,other)) != 1:
            return None
        return 'original' if original else 'self' if own else 'other'

    def _handover_relationship_v1(self, text):
        import re
        value = self._normalize_confirmation_text(text)
        if re.search(r'\b(?:not|don t|do not)\b', value):
            return None
        # Familiar relations plus an open description such as "her support worker".
        roles = (r'family member|boy friend|girl friend|boyfriend|girlfriend|friend|partner|'
                 r'husband|wife|spouse|mother|father|parent|son|daughter|child|brother|sister|'
                 r'sibling|cousin|aunt|uncle|grandmother|grandfather|grandparent|grandchild|'
                 r'nephew|niece|relative|caregiver|carer|guardian|representative|colleague|'
                 r'neighbour|neighbor|support worker|social worker')
        matched = re.search(r'\b(?:'+roles+r')\b',value)
        if matched:
            # A question about someone else's role is not the caller's answer.
            if re.search(r'\b(?:who is|are you|is he|is she|what is)\b',value):
                return None
            return matched.group(0)
        matched = re.search(r'\b(?:i am|i m|im) (?:his|her|their|the patient s) ([a-z]+(?: [a-z]+){0,4})',value)
        if matched:
            description = re.split(r'\b(?:and|but|can|could|would|please)\b',matched.group(1))[0].strip()
            if description and description not in {'yes','no','okay','ok'}:
                return description
        return None

    def _handover_reply_v1(self, text, answer):
        self.current_patient_text = text
        self.messages.append({'role':'user','content':text})
        self.messages.append({'role':'assistant','content':answer})
        return answer

    def _handover_set_subject_v1(self, subject):
        context = dict(self.session_state.get('handover_context_v1') or {})
        if subject != 'original':
            # Reuse the reviewed reset so previous names, reasons, selected
            # appointments and pending actions cannot cross patient subjects.
            self.session_state['speaker_change_pending'] = {'original_patient_name':self.session_state.get('patient_name')}
            self._process_before_lookup_repair('for myself' if subject=='self' else 'someone else')
        self.session_state['speaker_change_pending'] = None
        context['subject'] = subject
        self.session_state['handover_context_v1'] = context
        self.messages.append({'role':'system','content':
            'The caller stated the patient subject: '+subject+'. This is conversational clarification, '
            'not identity or access authorization. Keep name verification and separate transaction confirmation.'})

    def _handover_subject_gate_v1(self):
        context = self.session_state.get('handover_context_v1')
        if self.session_state.get('speaker_change_pending') or (context and not context.get('subject')):
            return {'success':False,'requires_patient_subject':True,'requires_patient_subject_clarification':True,
                    'reason':'Is this appointment for the original patient, for you, or for someone else?',
                    'instruction':'Ask whose appointment this is once; do not reveal existing appointment details or write until clarified.'}
        return None

    def find_booked_appointments(self, patient_name):
        return self._handover_subject_gate_v1() or self._find_before_relationship_v1(patient_name)

    def book_appointment(self, patient_name, slot_id, confirmed=False,
                         consultation_mode=None, insurance_status=None,
                         certificate_purpose=None, requested_date_range=None, reason_for_visit=None):
        gate = self._handover_subject_gate_v1()
        if gate:
            return gate
        return self._book_before_relationship_v1(patient_name,slot_id,confirmed,
            consultation_mode,insurance_status,certificate_purpose,requested_date_range,reason_for_visit)

    def _prepare_transaction_confirmation(self, action, arguments, details, confirmed):
        return self._handover_subject_gate_v1() or self._prepare_before_relationship_v1(action,arguments,details,confirmed)

    def process_turn(self, patient_text, speaker_assessment=None):
        import re
        text = str(patient_text or '').strip()
        fresh = isinstance(speaker_assessment,dict) and speaker_assessment.get('possible_change')
        if fresh:
            answer = self._process_before_relationship_v1(text,speaker_assessment)
            if self.session_state.get('speaker_change_pending'):
                self.session_state['handover_context_v1'] = {'relationship':None,'subject':None}
                self.session_state['pending_action'] = None
                answer = 'Am I speaking with someone else now? May I ask your relationship to the patient?'
                if self.messages and self.messages[-1].get('role')=='assistant':
                    self.messages[-1]['content'] = answer
            return answer
        pending = self.session_state.get('speaker_change_pending')
        if pending:
            relationship = self._handover_relationship_v1(text)
            subject = self._handover_subject_v1(text)
            if not relationship and not subject:
                return self._handover_reply_v1(text,'May I ask your relationship to the patient, or whether this request is for you?')
            self.session_state['handover_context_v1'] = {'relationship':relationship,'subject':None}
            self.session_state['speaker_change_pending'] = None
            self.session_state['pending_action'] = None
            if subject:
                self._handover_set_subject_v1(subject)
            self.messages.append({'role':'system','content':
                'A supported voice handover was acknowledged. The caller stated their relationship or patient subject. '
                'This does not prove identity. Answer general enquiries naturally. Clarify whose appointment only '
                'before a patient lookup or transaction if still unclear. Do not repeat the relationship question.'})
            return self._handover_reply_v1(text,
                'Thank you for letting me know. You can continue with your questions or request, on behalf of the patient or for yourself.')
        context = self.session_state.get('handover_context_v1')
        if context:
            subject = self._handover_subject_v1(text)
            if subject and subject != context.get('subject'):
                self._handover_set_subject_v1(subject)
                # Patient-subject clarification alone is not a transaction yes.
                if self._normalize_confirmation_text(text) in {'for me','for myself','for her','for him','for them','original patient','same patient','for the original patient','for the patient','someone else'}:
                    return self._handover_reply_v1(text,'Thank you. What would you like to do with that patient’s appointment?')
            if not self.session_state['handover_context_v1'].get('subject'):
                normalized = self._normalize_confirmation_text(text)
                # Existing-booking lookup requires a subject before any patient details.
                lookup = bool(re.search(r'\b(?:reschedule|cancel|cancellation|find my|look up|check my)\b',normalized))
                if lookup:
                    return self._handover_reply_v1(text,self._handover_subject_gate_v1()['reason'])
        reply = self._confirmed_booking_turn_v1(text)
        if reply is not None:
            return reply
        return self._process_before_relationship_v1(text,speaker_assessment)


    def _recover_taken_slot_v1(self, result, taken_slot):
        # The atomic RPC remains the authority; never retry another slot automatically.
        self.session_state['pending_action'] = None
        self.session_state['selected_slot_id'] = None
        self.session_state['nameflow_booking_arguments'] = None
        self.session_state['last_available_slots'] = []
        preferences = self.session_state.get('availability_preferences_v1')
        recovered = dict(result, requires_fresh_slot_selection=True, available_slots=[])
        if not preferences or preferences.get('service_name') != taken_slot.get('service_name'):
            recovered['instruction'] = 'That slot is no longer available. Please provide your consultant preference so I can check alternatives.'
            return recovered
        try:
            fresh = self.check_service_availability(**preferences)
        except Exception:
            self.session_state['last_available_slots'] = []
            recovered['availability_refresh_failed'] = True
            return recovered
        if fresh.get('success'):
            alternatives = [slot for slot in fresh.get('available_slots', [])
                            if slot.get('slot_id') != taken_slot.get('slot_id')]
            self.session_state['last_available_slots'] = alternatives
            recovered['available_slots'] = alternatives
        else:
            self.session_state['last_available_slots'] = []
            recovered['availability_refresh_failed'] = True
        return recovered

    def _booking_result_text_v1(self, result, details=None):
        if result.get('requires_confirmation'):
            return self._booking_confirmation_text_v1(result['confirmation_details'])
        if result.get('success') is True:
            d = dict(details or {})
            d.update({k:v for k,v in result.items() if v is not None})
            # The selected offered slot supplies presentation details if the RPC omits them.
            slot = next((s for s in self.session_state.get('last_available_slots', [])
                         if s.get('slot_id') == result.get('slot_id', self.session_state.get('selected_slot_id'))), {})
            d = dict(slot, **d)
            if all(d.get(k) for k in ('patient_name','provider','service_name','display_date','display_time')):
                answer = (f"The appointment for {d['patient_name']} with {d['provider']} for "
                          f"{d['service_name']} on {d['display_date']} at {d['display_time']} is booked.")
            else:
                answer = 'The appointment has been booked successfully.'
            if result.get('appointment_id') is not None:
                answer += f" The appointment ID is {result['appointment_id']}."
            if result.get('intake_warning'):
                answer += ' The appointment is booked, but the additional certificate information could not be saved.'
            return answer + ' Is there anything else I can help you with?'
        if result.get('requires_fresh_slot_selection'):
            slots = result.get('available_slots') or []
            answer = 'That slot is no longer available.'
            if slots:
                lines = [f"- {s['provider']} on {s['display_date']} at {s['display_time']}" for s in slots[:3]]
                return answer + ' These alternatives are available:\n' + '\n'.join(lines) + '\nWhich would you prefer?'
            if result.get('instruction'):
                return result['instruction']
            if result.get('availability_refresh_failed'):
                return answer + ' I could not check alternatives just now. Please try checking availability again.'
            return answer + ' There are currently no other slots matching your preferences. Would you like to check another preference?'
        return str(result.get('reason') or result.get('instruction') or 'The booking was not completed.')

    def _confirmed_booking_turn_v1(self, text):
        import re
        pending = self.session_state.get('pending_action')
        if not isinstance(pending, dict) or pending.get('action') != 'book':
            return None
        if self.session_state.get('nameflow_v1') or self.session_state.get('lookup_nameflow') or self.session_state.get('speaker_change_pending'):
            return None
        if self._handover_subject_gate_v1():
            return None
        normalized = self._normalize_confirmation_text(text)
        # Accept only assent with polite filler. Changes and questions stay in the normal path.
        polite = normalized.split()
        confirmation = (len(polite) <= 12 and set(polite) <= {'yes','yeah','yep','yup','ok','okay','alright','sure','certainly','absolutely','definitely','confirm','confirmed','correct','agreed','fine','please','all','good','sounds','great','that','this','is','s','works','work','for','me','go','ahead','proceed','do','it','on','then','perfect','thank','thanks','you','so','much','i','book','booking','appointment','the'} and (self._is_explicit_confirmation(text, 'book') or
                        (len(polite) <= 8 and bool(set(polite) & {'yes','yeah','yep','yup'}) and
                         set(polite) <= {'yes','yeah','yep','yup','please','perfect','thank','thanks','you','so','much'})))
        if confirmation:
            args = pending.get('arguments') or {}
            verified = self.session_state.get('verified_patient_name')
            if not verified or self._name_letters(verified) != self._name_letters(args.get('patient_name')):
                return None
            self.current_patient_text = text
            self.messages.append({'role':'user','content':text})
            allowed = ('patient_name','slot_id','consultation_mode','insurance_status','certificate_purpose','requested_date_range','reason_for_visit')
            arguments = {k:v for k,v in args.items() if k in allowed}
            try:
                result = self.book_appointment(confirmed=True, **arguments)
            except Exception:
                self.session_state['pending_action'] = None
                answer = 'I could not confirm the booking result. Please check your appointment before trying again.'
                self.messages.append({'role':'assistant','content':answer})
                return answer
            # Preserve an inspectable tool result without inventing another model tool call.
            print('CONFIRMED BOOKING RESULT: ' + json.dumps(result, default=str))
            self.messages.append({'role':'system','content':'Server booking result: '+json.dumps(result, default=str)})
            answer = self._booking_result_text_v1(result, pending.get('details'))
            self.messages.append({'role':'assistant','content':answer})
            return answer
        if re.search(r'\b(?:bye|goodbye)\b', normalized) and not re.search(r'\b(?:book|confirm|proceed|go ahead)\b', normalized):
            self.session_state['pending_action'] = None
            self.session_state['nameflow_booking_arguments'] = None
            return self._nameflow_reply(text, 'Goodbye. The pending booking has not been completed.')
        return None













    # Same-call correction: retain only a booking successfully created by this
    # session. A reference number from a different call is not authorization.
    _book_before_same_call_correction = book_appointment
    _turn_before_same_call_correction = process_turn

    def book_appointment(self, patient_name, slot_id, confirmed=False,
                         consultation_mode=None, insurance_status=None,
                         certificate_purpose=None, requested_date_range=None,
                         reason_for_visit=None):
        result = self._book_before_same_call_correction(
            patient_name, slot_id, confirmed, consultation_mode, insurance_status,
            certificate_purpose, requested_date_range, reason_for_visit)
        if isinstance(result, dict) and result.get('success') is True and result.get('appointment_id') is not None:
            self.session_state['same_call_booking_v1'] = {
                'appointment_id': result['appointment_id'],
                'patient_name': result.get('patient_name') or patient_name,
                'slot_id': result.get('slot_id', slot_id),
            }
            self.session_state['same_call_name_correction_v1'] = None
        return result

    def _clear_letter_readback_v1(self, parts):
        labels = ['First name', 'Last name'] if len(parts) == 2 else [
            'Name part ' + str(i + 1) for i in range(len(parts))]
        return ' '.join(label + ': ' + part.capitalize() + '. Letters: ' +
                        ', '.join(part.upper()) + '.'
                        for label, part in zip(labels, parts))

    def _nameflow_readback(self, flow):
        name = ' '.join(part.capitalize() for part in flow['parts'])
        flow.update(stage='readback', candidate=name)
        self.session_state['pending_name_verification'] = {
            'patient_name': name, 'slot_id': (flow.get('booking') or {}).get('slot_id')}
        return (self._clear_letter_readback_v1(flow['parts']) +
                ' Is that the complete patient name, with every letter correct?')

    def _lookup_readback(self, flow):
        name = ' '.join(part.capitalize() for part in flow['parts'])
        flow.update(stage='readback', candidate=name)
        return (self._clear_letter_readback_v1(flow['parts']) +
                ' Is that the complete patient name, with every letter correct?')

    def _correction_assent_v1(self, text):
        value = self._normalize_confirmation_text(text)
        words = value.split()
        allowed = {'yes', 'yeah', 'yep', 'yup', 'correct', 'confirm', 'confirmed',
                   'please', 'thank', 'thanks', 'you', 'so', 'much', 'perfect',
                   'that', 'is', 'right', 'the', 'name', 'spelling', 'update',
                   'it', 'go', 'ahead', 'all', 'good'}
        return bool(words and len(words) <= 12 and set(words) <= allowed and
                    set(words) & {'yes', 'yeah', 'yep', 'yup', 'correct', 'confirm', 'confirmed'})

    def _correction_candidate_v1(self, flow, parts):
        flow['parts'] = parts
        flow['candidate'] = ' '.join(part.capitalize() for part in parts)
        flow['stage'] = 'confirm'
        return (self._clear_letter_readback_v1(parts) +
                ' Do you confirm updating the name on appointment ' +
                str(flow['booking']['appointment_id']) +
                '? Its date and time will stay the same.')

    def _commit_same_call_name_correction_v1(self, flow):
        old = flow['booking']
        candidate = flow['candidate']
        # Compare-and-set protects against concurrent rescheduling, cancellation
        # or name changes. Update one appointment, not every matching name.
        try:
            response = (supabase.table('appointments')
                        .update({'patient_name': candidate})
                        .eq('appointment_id', old['appointment_id'])
                        .eq('patient_name', old['patient_name'])
                        .eq('slot_id', old['slot_id'])
                        .eq('status', 'booked').execute())
            rows = response.data
        except Exception:
            return ('I could not verify whether the correction was saved. '
                    'Please check appointment ' + str(old['appointment_id']) +
                    ' before trying again; I will not repeat the update automatically.')
        if (not isinstance(rows, list) or len(rows) != 1 or
                rows[0].get('appointment_id') != old['appointment_id'] or
                rows[0].get('patient_name') != candidate or
                rows[0].get('slot_id') != old['slot_id'] or
                rows[0].get('status') != 'booked'):
            return ('I could not verify the name correction. The appointment may '
                    'have changed. Please check its current details before trying again.')
        parts = list(flow['parts'])
        key = self._name_letters(candidate)
        self.session_state.update(
            patient_name=candidate, verified_patient_name=candidate,
            verified_patient_name_key=key, observed_spelled_name_key=key,
            observed_spelled_name_parts=parts, observed_spelled_name_confirmed=True,
            pending_name_verification=None, pending_action=None,
            nameflow_v1=None, nameflow_booking_arguments=None, lookup_nameflow=None,
            lookup_verified_name=candidate, lookup_operation=None,
            active_appointment_id=old['appointment_id'], selected_slot_id=old['slot_id'])
        self.session_state['same_call_booking_v1'] = dict(old, patient_name=candidate)
        for item in self.session_state.get('last_found_appointments', []):
            if item.get('appointment_id') == old['appointment_id']:
                item['patient_name'] = candidate
        result = {'success': True, 'appointment_id': old['appointment_id'],
                  'old_patient_name': old['patient_name'], 'patient_name': candidate}
        print('CONFIRMED SAME-CALL NAME CORRECTION: ' + json.dumps(result))
        self.messages.append({'role': 'system', 'content':
            'The server updated only the patient_name on appointment ' +
            str(old['appointment_id']) + ' to ' + candidate +
            '. Previous spelling is superseded. Date and time were unchanged.'})
        return ('The name on appointment ' + str(old['appointment_id']) +
                ' has been corrected to ' + candidate +
                '. The appointment date and time are unchanged. How else can I help?')

    def _same_call_name_turn_v1(self, text):
        import re
        normalized = self._normalize_confirmation_text(text)
        booking = self.session_state.get('same_call_booking_v1')
        flow = self.session_state.get('same_call_name_correction_v1')
        requested = bool(re.search(r'\b(?:name|surname|spelling)\b', normalized) and
                         re.search(r'\b(?:correct|correction|fix|rectify|wrong|incorrect|mistake|misspelled|change|update)\b', normalized))
        if not flow and not requested:
            return None
        # New patients, unresolved handovers, or a different pending operation
        # must never inherit permission to rename the previous patient's booking.
        gate = self._handover_subject_gate_v1()
        verified = self.session_state.get('verified_patient_name')
        if (not booking or self.session_state.get('nameflow_v1') or
                self.session_state.get('lookup_nameflow') or
                self._name_letters(verified) != self._name_letters(booking['patient_name'])):
            self.session_state['same_call_name_correction_v1'] = None
            return None
        if gate:
            self.session_state['same_call_name_correction_v1'] = None
            return self._lookup_reply(text, gate['reason'])
        if re.search(r'\b(?:someone else|another patient|different patient|different person)\b', normalized):
            self.session_state['same_call_name_correction_v1'] = None
            self.session_state['same_call_booking_v1'] = None
            return None
        if flow and (normalized in {'stop', 'stop please', 'never mind', 'cancel this request',
                                    'do not update', 'don t update'} or
                     re.search(r'\b(?:cancel|reschedule|move)\b.*\b(?:appointment|booking)\b', normalized)):
            self.session_state['same_call_name_correction_v1'] = None
            if normalized in {'stop', 'stop please', 'never mind', 'cancel this request', 'do not update', 'don t update'}:
                return self._lookup_reply(text, 'I have stopped the name correction. No name update was made.')
            return None
        if flow is None:
            pending = self.session_state.get('pending_action')
            if pending:
                return self._lookup_reply(text, 'Please finish or stop the pending appointment request before correcting the name.')
            flow = {'booking': dict(booking), 'stage': 'collect'}
            self.session_state['same_call_name_correction_v1'] = flow
        groups = self._nameflow_groups(text)
        if groups:
            original_parts = booking['patient_name'].split()
            if 'last name' in normalized or 'surname' in normalized:
                if len(groups) != 1 or len(original_parts) < 2:
                    answer = 'Please spell the complete corrected name, with a pause between its parts.'
                else:
                    answer = self._correction_candidate_v1(flow, original_parts[:-1] + groups)
            elif 'first name' in normalized:
                if len(groups) != 1:
                    answer = 'Please spell just the corrected first name, or say full name and spell every part.'
                else:
                    answer = self._correction_candidate_v1(flow, groups + original_parts[1:])
            elif len(groups) >= 2 or 'full name' in normalized or 'complete name' in normalized:
                answer = self._correction_candidate_v1(flow, groups)
            else:
                flow.update(stage='which_part', supplied_parts=groups)
                answer = ('You spelled ' + ', '.join(groups[0].upper()) +
                          '. Is that your first name, last name, or complete name?')
            return self._lookup_reply(text, answer)
        if flow['stage'] == 'which_part':
            original_parts = booking['patient_name'].split()
            parts = flow['supplied_parts']
            if 'last name' in normalized or 'surname' in normalized:
                if len(original_parts) < 2:
                    return self._lookup_reply(text, 'Please spell the complete corrected name.')
                answer = self._correction_candidate_v1(flow, original_parts[:-1] + parts)
            elif 'first name' in normalized:
                answer = self._correction_candidate_v1(flow, parts + original_parts[1:])
            elif 'full name' in normalized or 'complete name' in normalized:
                answer = self._correction_candidate_v1(flow, parts)
            else:
                answer = 'Is the spelling you gave for your first name, last name, or complete name?'
            return self._lookup_reply(text, answer)
        if flow['stage'] == 'confirm':
            if self._correction_assent_v1(text):
                # Consume before writing. A repeated yes cannot replay this update.
                self.session_state['same_call_name_correction_v1'] = None
                answer = self._commit_same_call_name_correction_v1(flow)
                return self._lookup_reply(text, answer)
            if self._is_explicit_rejection(text):
                flow['stage'] = 'collect'
                return self._lookup_reply(text, 'Please spell the part that is wrong and say first name, last name, or full name.')
            return self._lookup_reply(text, 'Please confirm the exact name correction I read back, or spell the part you want to change.')
        return self._lookup_reply(text, 'I can correct the name on the appointment just booked. Please spell the incorrect part and say first name, last name, or full name.')

    def process_turn(self, patient_text, speaker_assessment=None):
        text = str(patient_text or '').strip()
        change = isinstance(speaker_assessment, dict) and (
            speaker_assessment.get('possible_change') or
            speaker_assessment.get('returning_voice_candidate'))
        if change or self.session_state.get('speaker_change_pending'):
            self.session_state['same_call_name_correction_v1'] = None
            self.session_state['same_call_booking_v1'] = None
            return self._turn_before_same_call_correction(text, speaker_assessment)
        reply = self._same_call_name_turn_v1(text)
        if reply is not None:
            return reply
        return self._turn_before_same_call_correction(text, speaker_assessment)



    # Reference management is deterministic. The model cannot authorize writes
    # by inventing an ID, a name confirmation, or a speaker relationship.
    _turn_before_reference_v2 = process_turn
    _reschedule_before_reference_v2 = reschedule_appointment
    _cancel_before_reference_v2 = cancel_appointment

    def _reference_clear_v2(self, forget_patient=False):
        for key in ('reference_flow_v2', 'reference_verified_v2', 'same_call_booking_v1',
                    'same_call_name_correction_v1', 'pending_action', 'lookup_nameflow',
                    'lookup_operation', 'nameflow_v1', 'nameflow_booking_arguments',
                    'pending_name_verification', 'active_appointment_id', 'selected_slot_id'):
            self.session_state[key] = None
        self.session_state['last_found_appointments'] = []
        self.session_state['last_available_slots'] = []
        if forget_patient:
            for key in ('patient_name', 'verified_patient_name', 'verified_patient_name_key',
                        'observed_spelled_name_key', 'lookup_verified_name'):
                self.session_state[key] = None
            self.session_state['observed_spelled_name_parts'] = []
            self.session_state['observed_spelled_name_confirmed'] = False
            self.messages = [m for m in self.messages[:1] if m.get('role') == 'system']

    def _reference_id_v2(self, text, bare=False):
        import re
        value = self._normalize_confirmation_text(text)
        pattern = r'\b(?:appointment|booking|reference)(?:\s+(?:id|number|reference))?\s*(?:is\s*|number\s*|id\s*)?(\d+)\b'
        match = re.search(pattern, value)
        if match:
            return int(match.group(1)) if int(match.group(1)) > 0 else None
        words = {'zero':'0','one':'1','two':'2','three':'3','four':'4','five':'5',
                 'six':'6','seven':'7','eight':'8','nine':'9'}
        cleaned = re.sub(r'\b(?:my|the|appointment|booking|reference|id|number|is|it|its)\b', '', value).split()
        if cleaned and all(w in words for w in cleaned):
            number = int(''.join(words[w] for w in cleaned))
            return number if number > 0 else None
        if bare and re.fullmatch(r'\d+', value):
            return int(value) if int(value) > 0 else None
        # Natural spoken tens, without interpreting dates or slot numbers as IDs.
        tens = {'ten':10,'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,
                'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,'nineteen':19,
                'twenty':20,'thirty':30,'forty':40,'fifty':50,'sixty':60,'seventy':70,
                'eighty':80,'ninety':90}
        if len(cleaned) == 1 and cleaned[0] in tens:
            return tens[cleaned[0]]
        if len(cleaned) == 2 and cleaned[0] in tens and tens[cleaned[0]] >= 20 and cleaned[1] in words:
            return tens[cleaned[0]] + int(words[cleaned[1]])
        return None

    def _reference_operation_v2(self, text):
        import re
        value = self._normalize_confirmation_text(text)
        if re.search(r'\b(?:cancel|cancellation)\b', value):
            return 'cancel'
        if re.search(r'\b(?:reschedule|re schedule|move|change|update)\b', value) and re.search(r'\b(?:date|time|appointment|booking|reschedule|re schedule)\b', value) and not re.search(r'\b(?:name|surname|spelling)\b', value):
            return 'reschedule'
        if re.search(r'\b(?:name|surname|spelling)\b', value) and re.search(r'\b(?:correct|correction|fix|rectify|wrong|incorrect|mistake|misspelled|change|update)\b', value):
            return 'name'
        if re.search(r'\b(?:previous|existing|last|already booked|look up|lookup|reference|appointment id|booking id|check my booking|check my appointment)\b', value):
            return 'details'
        return None

    def _reference_record_v2(self, appointment_id):
        # Base tables, not an assumed view or new schema.
        rows = (supabase.table('appointments').select('appointment_id,patient_name,doctor_id,slot_id,status')
                .eq('appointment_id', appointment_id).limit(2).execute()).data
        if not isinstance(rows, list) or len(rows) != 1:
            return None
        row = dict(rows[0])
        slot = (supabase.table('slots').select('slot_id,doctor_id,service_id,appointment_date,appointment_time,consultation_mode')
                .eq('slot_id', row['slot_id']).single().execute()).data
        doctor = (supabase.table('doctors').select('doctor_id,doctor_name')
                  .eq('doctor_id', row['doctor_id']).single().execute()).data
        service = service_by_id.get(slot['service_id']) if slot else None
        if not slot or not doctor or not service or not row.get('patient_name'):
            raise ValueError('Incomplete appointment data')
        row.update(appointment_date=str(slot['appointment_date']), appointment_time=str(slot['appointment_time']),
                   consultation_mode=slot.get('consultation_mode') or 'in_person',
                   provider=doctor['doctor_name'], service_name=service['service_name'],
                   display_date=format_date(slot['appointment_date']), display_time=format_time(slot['appointment_time']))
        return row

    def _reference_same_record_v2(self, before, after):
        return bool(after and all(before.get(k) == after.get(k) for k in
                    ('appointment_id','patient_name','slot_id','doctor_id','status')))

    def _reference_date_v2(self, text):
        import re
        from datetime import datetime, timedelta
        value = str(text).lower().replace(',', ' ')
        value = re.sub(r'(\d+)(?:st|nd|rd|th)\b', r'\1', value)
        for pattern, formats in (
            (r'\b\d{4}-\d{1,2}-\d{1,2}\b', ('%Y-%m-%d',)),
            (r'\b\d{1,2}/\d{1,2}/\d{4}\b', ('%d/%m/%Y',)),
            (r'\b\d{1,2}\s+[a-z]+\s+\d{4}\b', ('%d %B %Y','%d %b %Y')),
            (r'\b[a-z]+\s+\d{1,2}\s+\d{4}\b', ('%B %d %Y','%b %d %Y'))):
            for match in re.finditer(pattern, value):
                for fmt in formats:
                    try:
                        return datetime.strptime(match.group(), fmt).date().isoformat()
                    except ValueError:
                        pass
        if 'tomorrow' in value:
            return (sydney_now().date() + timedelta(days=1)).isoformat()
        if re.search(r'\btoday\b', value):
            return sydney_now().date().isoformat()
        return None

    def _reference_time_v2(self, text):
        import re
        value = str(text).lower().replace('a.m.', 'am').replace('p.m.', 'pm')
        match = re.search(r'\b(\d{1,2})(?::(\d{2})|\s+(\d{2}))?\s*(am|pm)\b', value)
        if match:
            hour, minute = int(match[1]), int(match[2] or match[3] or 0)
            if not 1 <= hour <= 12 or minute >= 60:
                return None
            return '%02d:%02d' % (hour % 12 + (12 if match[4] == 'pm' else 0), minute)
        match = re.search(r'\b(\d{1,2}):(\d{2})\b', value)
        if match and int(match[1]) < 24 and int(match[2]) < 60:
            return '%02d:%02d' % (int(match[1]), int(match[2]))
        return None

    def _reference_activate_v2(self, record):
        self.session_state['reference_verified_v2'] = dict(record)
        self.session_state['last_found_appointments'] = [dict(record)]
        self.session_state.update(patient_name=record['patient_name'],
            active_appointment_id=record['appointment_id'], active_service=record['service_name'],
            selected_slot_id=record['slot_id'], lookup_nameflow=None, lookup_operation=None,
            pending_name_verification=None, nameflow_v1=None, pending_action=None)

    def _reference_guard_v2(self, appointment_id):
        record = self.session_state.get('reference_verified_v2')
        if self.session_state.get('speaker_hold_v2') or self._handover_subject_gate_v1() or not record or record['appointment_id'] != appointment_id:
            return {'success':False,'reason':'Please provide the appointment reference and verify its details before making a change.'}
        try:
            current = self._reference_record_v2(appointment_id)
        except Exception:
            return {'success':False,'reason':'I could not check the current appointment. No change was requested.'}
        if not self._reference_same_record_v2(record, current) or current['status'] != 'booked':
            self._reference_clear_v2()
            return {'success':False,'reason':'The appointment has changed. Please provide the reference again so I can check its current details.'}
        return None

    def find_booked_appointments(self, patient_name):
        record = self.session_state.get('reference_verified_v2')
        if self._handover_subject_gate_v1() or not record:
            return {'success':False,'requires_booking_reference':True,
                    'reason':'Ask for the appointment ID or booking reference, not name spelling, to locate an existing booking.'}
        return {'success':True,'patient_name':record['patient_name'],'booked_appointments':[dict(record)]}

    def reschedule_appointment(self, appointment_id, new_slot_id, confirmed=False):
        gate = self._reference_guard_v2(appointment_id)
        if gate:
            return gate
        return self._reschedule_before_reference_v2(appointment_id, new_slot_id, confirmed)

    def cancel_appointment(self, appointment_id, confirmed=False):
        gate = self._reference_guard_v2(appointment_id)
        if gate:
            return gate
        return self._cancel_before_reference_v2(appointment_id, confirmed)

    def _reference_summary_v2(self, record):
        return ('Appointment ' + str(record['appointment_id']) + ' for ' + record['patient_name'] +
                ' with ' + record['provider'] + ' for ' + record['service_name'] + ' on ' +
                record['display_date'] + ' at ' + record['display_time'])

    def _reference_begin_action_v2(self, text, flow):
        record = flow['record']
        if record['status'] != 'booked':
            flow['stage'] = 'ready'
            return self._reference_summary_v2(record) + ' is ' + str(record['status']) + '. It cannot be modified as a current booking.'
        operation = flow['operation']
        if operation == 'name':
            # Reuse the reviewed, confirmed compare-and-set name updater.
            self.session_state['same_call_booking_v1'] = dict(record)
            self.session_state['verified_patient_name'] = record['patient_name']
            self.session_state['verified_patient_name_key'] = self._name_letters(record['patient_name'])
            flow['stage'] = 'name'
            return 'I have located appointment ' + str(record['appointment_id']) + '. Please spell only the part to correct and say first name, last name, or full name.'
        if operation == 'cancel':
            result = self.cancel_appointment(record['appointment_id'], confirmed=False)
            if not result.get('requires_confirmation'):
                return str(result.get('reason') or 'Cancellation was not prepared.')
            flow['stage'] = 'confirm_cancel'
            return self._reference_summary_v2(record) + '. Do you confirm cancelling this appointment?'
        if operation == 'reschedule':
            flow['stage'] = 'preference'
            return 'What new date would you prefer? Please include the year. You can also say next available.'
        flow['stage'] = 'ready'
        return self._reference_summary_v2(record) + ' is ' + str(record['status']) + '. Would you like to correct the name, reschedule, or cancel it?'

    def _reference_offer_v2(self, text, flow):
        record = flow['record']
        date = self._reference_date_v2(text)
        normalized = self._normalize_confirmation_text(text)
        if not date and not any(cue in normalized for cue in ('next available','earliest','any date')):
            return 'Please give the new date, including the year, for example 14 October 2026, or say next available.'
        try:
            # Query the requested date directly: do not mistake a bounded earliest
            # availability list for all available dates.
            service = next((s for s in service_by_id.values() if s['service_name'] == record['service_name']), None)
            query = (supabase.table('slots').select('slot_id,doctor_id,service_id,appointment_date,appointment_time,available,consultation_mode')
                     .eq('service_id', service['service_id']).eq('available', True)
                     .eq('consultation_mode', record['consultation_mode']))
            query = query.eq('appointment_date', date) if date else query.gte('appointment_date', sydney_now().date().isoformat())
            rows = query.order('appointment_date').order('appointment_time').limit(50).execute().data or []
            slots = []
            seen = set()
            for row in rows:
                if not slot_is_in_future(row):
                    continue
                doctor = (supabase.table('doctors').select('doctor_id,doctor_name').eq('doctor_id', row['doctor_id']).single().execute()).data
                key = (row['doctor_id'], str(row['appointment_date']), str(row['appointment_time']))
                if key in seen or not doctor:
                    continue
                seen.add(key)
                slot = dict(row, provider=doctor['doctor_name'], service_name=record['service_name'],
                            display_date=format_date(row['appointment_date']), display_time=format_time(row['appointment_time']))
                slots.append(slot)
                if len(slots) >= 5:
                    break
        except Exception:
            return 'I could not check available times just now. Your existing appointment is unchanged. Please try the date again.'
        if not slots:
            return 'I found no future available times in that check. Your existing appointment is unchanged. Please give another date or say next available.'
        flow.update(stage='choose_slot', slots=slots)
        self.session_state['last_available_slots'] = slots
        return 'Available options: ' + ' '.join('Option ' + str(i + 1) + ': ' + s['provider'] + ' on ' + s['display_date'] + ' at ' + s['display_time'] + '.' for i,s in enumerate(slots)) + ' Which option would you prefer?'

    def _reference_choose_v2(self, text, flow):
        import re
        value = self._normalize_confirmation_text(text)
        options = {'first':1,'second':2,'third':3,'fourth':4,'fifth':5,'one':1,'two':2,'three':3,'four':4,'five':5}
        match = re.fullmatch(r'(?:option |the |number )?(first|second|third|fourth|fifth|one|two|three|four|five|[1-5])(?: option| one| please)?', value)
        index = (options.get(match[1]) or int(match[1])) if match else None
        if index and index <= len(flow['slots']):
            selected = flow['slots'][index-1]
        else:
            date, time = self._reference_date_v2(text), self._reference_time_v2(text)
            matches = [s for s in flow['slots'] if time and str(s['appointment_time'])[:5] == time and (not date or s['appointment_date'] == date)]
            if len(matches) != 1:
                return 'Please say the option number so I can select the exact offered appointment.'
            selected = matches[0]
        result = self.reschedule_appointment(flow['record']['appointment_id'], selected['slot_id'], confirmed=False)
        if not result.get('requires_confirmation'):
            return str(result.get('reason') or 'Rescheduling was not prepared.')
        flow.update(stage='confirm_reschedule', selected=dict(selected))
        return ('Do you confirm moving appointment ' + str(flow['record']['appointment_id']) + ' for ' + flow['record']['patient_name'] +
                ' from ' + flow['record']['display_date'] + ' at ' + flow['record']['display_time'] +
                ' to ' + selected['provider'] + ' on ' + selected['display_date'] + ' at ' + selected['display_time'] + '?')

    def _reference_commit_v2(self, flow):
        record = flow['record']
        # Consume the conversational yes before contacting the database. Existing
        # atomic RPCs decide slot availability and release/claim slots together.
        operation = 'cancel' if flow['stage'] == 'confirm_cancel' else 'reschedule'
        flow['stage'] = 'result_pending'
        try:
            result = (self.cancel_appointment(record['appointment_id'], confirmed=True) if operation == 'cancel' else
                      self.reschedule_appointment(record['appointment_id'], flow['selected']['slot_id'], confirmed=True))
            self.session_state['pending_action'] = None
            if not isinstance(result, dict) or result.get('success') is not True:
                flow['stage'] = 'ready'
                return str((result or {}).get('reason') if isinstance(result,dict) else '') or 'The change was not confirmed. Please check the current booking before trying again.'
            current = self._reference_record_v2(record['appointment_id'])
            expected_status = 'cancelled' if operation == 'cancel' else 'booked'
            expected_slot = record['slot_id'] if operation == 'cancel' else flow['selected']['slot_id']
            if not current or current['status'] != expected_status or current['slot_id'] != expected_slot:
                raise ValueError('Cannot verify committed state')
        except Exception:
            self.session_state['pending_action'] = None
            self.session_state['reference_flow_v2'] = None
            self.session_state['reference_verified_v2'] = None
            return 'I could not verify the final result for appointment ' + str(record['appointment_id']) + '. Please check that reference before repeating the request; I will not retry automatically.'
        print('CONFIRMED REFERENCE BOOKING CHANGE: ' + json.dumps({'success':True,'action':operation,'appointment_id':record['appointment_id'],'slot_id':current['slot_id'],'status':current['status']}))
        flow.update(stage='ready', record=current, operation='details')
        self._reference_activate_v2(current)
        self.session_state['same_call_booking_v1'] = dict(current) if operation == 'reschedule' else None
        self.messages.append({'role':'system','content':'Server confirmed '+operation+' of appointment '+str(record['appointment_id'])+'. Supersede prior date/time/status.'})
        return self._reference_summary_v2(current) + (' has been cancelled.' if operation == 'cancel' else ' has been rescheduled.') + ' How else can I help?'

    def _reference_turn_v2(self, text):
        import re
        value = self._normalize_confirmation_text(text)
        flow = self.session_state.get('reference_flow_v2')
        operation = self._reference_operation_v2(text)
        requested_id = self._reference_id_v2(text, bare=bool(flow and flow['stage'] == 'reference'))
        if not flow and not operation and requested_id is None:
            return None
        if value in {'stop','stop please','never mind','cancel this request','do not update','don t update','goodbye','bye'}:
            self._reference_clear_v2()
            return self._lookup_reply(text, 'I have stopped this change request. No further appointment change was requested.')
        subject_gate = self._handover_subject_gate_v1()
        if subject_gate:
            return self._lookup_reply(text, subject_gate['reason'])
        if flow is None:
            self.session_state['pending_action'] = None
            self.session_state['lookup_nameflow'] = None
            self.session_state['nameflow_v1'] = None
            flow = {'stage':'reference','operation':operation or 'details','attempts':0}
            self.session_state['reference_flow_v2'] = flow
            recent = self.session_state.get('same_call_booking_v1')
            # A successful booking in this uninterrupted patient context can be
            # managed immediately without making the patient repeat an ID.
            previous = bool(re.search(r'\b(?:previous|last|another|different|earlier)\b', value))
            if recent and not previous and (requested_id is None or requested_id == recent['appointment_id']):
                try:
                    current = self._reference_record_v2(recent['appointment_id'])
                except Exception:
                    current = None
                if current and self._reference_same_record_v2(dict(recent, doctor_id=current['doctor_id'], status='booked'), current):
                    self._reference_activate_v2(current)
                    flow['record'] = current
                    return self._lookup_reply(text, self._reference_begin_action_v2(text, flow))
        if requested_id is not None and (flow['stage'] == 'reference' or requested_id != (flow.get('record') or {}).get('appointment_id')):
            # New reference invalidates all previous transaction authority.
            keep_operation = operation or flow['operation']
            self._reference_clear_v2()
            flow = {'stage':'reference','operation':keep_operation,'attempts':0}
            self.session_state['reference_flow_v2'] = flow
            try:
                record = self._reference_record_v2(requested_id)
            except Exception:
                return self._lookup_reply(text, 'I could not check that reference just now. Please try it again. No appointment change was requested.')
            if not record:
                return self._lookup_reply(text, 'I could not verify that booking reference. Please check the appointment ID and try again.')
            flow.update(stage='verify_details', record=record)
            # The existing prototype has no authenticated caller-phone binding.
            # Independently supplied date/time is record matching, not proof of
            # identity. Do not reveal the stored name before these details match.
            return self._lookup_reply(text, 'Before I read the booking details, please tell me its currently booked date, including the year, and time, with AM or PM.')
        if flow['stage'] == 'reference':
            return self._lookup_reply(text, 'Please tell me the appointment ID or booking reference. This is the appointment number, not the slot number.')
        if flow['stage'] == 'verify_details':
            date, time = self._reference_date_v2(text), self._reference_time_v2(text)
            if date:
                flow['given_date'] = date
            if time:
                flow['given_time'] = time
            if not flow.get('given_date') or not flow.get('given_time'):
                return self._lookup_reply(text, 'Please give the currently booked date including the year, and the booked time with AM or PM. I need both to check the reference.')
            record = flow['record']
            if flow['given_date'] != record['appointment_date'] or flow['given_time'] != record['appointment_time'][:5]:
                flow['attempts'] += 1
                flow.pop('given_date',None); flow.pop('given_time',None)
                if flow['attempts'] >= 3:
                    self._reference_clear_v2()
                    return self._lookup_reply(text, 'I could not verify the booking details. No change was made. Please check your booking information before trying again.')
                return self._lookup_reply(text, 'Those details did not verify the reference. Please check the currently booked date and time and provide them again.')
            flow['stage'] = 'confirm_record'
            return self._lookup_reply(text, 'The stored patient name is ' + record['patient_name'] + '. Is this the booking you want to manage, even if its spelling needs correction?')
        if flow['stage'] == 'confirm_record':
            if self._correction_assent_v1(text):
                self._reference_activate_v2(flow['record'])
                return self._lookup_reply(text, self._reference_begin_action_v2(text, flow))
            if self._is_explicit_rejection(text):
                self._reference_clear_v2()
                return self._lookup_reply(text, 'I have not changed that booking. Please provide the correct appointment reference.')
            return self._lookup_reply(text, 'Is this the booking you want to manage? Please say yes or no. I will collect the corrected spelling separately.')
        if operation in {'name','cancel','reschedule'} and operation != flow['operation']:
            self.session_state['pending_action'] = None
            self.session_state['same_call_name_correction_v1'] = None
            flow['operation'] = operation
            return self._lookup_reply(text,self._reference_begin_action_v2(text,flow))
        if flow['stage'] == 'name':
            if operation in {'cancel','reschedule'}:
                self.session_state['same_call_name_correction_v1'] = None
                flow['operation'] = operation
                return self._lookup_reply(text, self._reference_begin_action_v2(text,flow))
            if not self.session_state.get('same_call_name_correction_v1'):
                # Start the spelling state machine, then process the actual turn.
                self._same_call_name_turn_v1('Please correct the name')
            gate = self._reference_guard_v2(flow['record']['appointment_id'])
            if gate:
                self.session_state['same_call_name_correction_v1'] = None
                return self._lookup_reply(text,gate['reason'])
            answer = self._same_call_name_turn_v1(text)
            if not self.session_state.get('same_call_name_correction_v1'):
                updated = self.session_state.get('same_call_booking_v1')
                if updated and updated['patient_name'] != flow['record']['patient_name']:
                    flow['record'] = dict(flow['record'], patient_name=updated['patient_name'])
                    self._reference_activate_v2(flow['record'])
                flow['stage'] = 'ready'
            return answer or self._lookup_reply(text,'Please spell only the part to correct and say first name, last name, or full name.')
        if flow['stage'] in {'confirm_cancel','confirm_reschedule'}:
            if self._correction_assent_v1(text):
                return self._lookup_reply(text,self._reference_commit_v2(flow))
            if self._is_explicit_rejection(text):
                self.session_state['pending_action'] = None
                flow['stage'] = 'ready'
                return self._lookup_reply(text,'I have not made that change. What would you like to do instead?')
            if operation and operation != flow['operation']:
                self.session_state['pending_action'] = None
                flow['operation'] = operation
                return self._lookup_reply(text,self._reference_begin_action_v2(text,flow))
            return self._lookup_reply(text,'Please confirm the exact change I read back, or say no to change your request.')
        if flow['stage'] == 'preference':
            return self._lookup_reply(text,self._reference_offer_v2(text,flow))
        if flow['stage'] == 'choose_slot':
            if operation in {'name','cancel'}:
                flow['operation'] = operation
                return self._lookup_reply(text,self._reference_begin_action_v2(text,flow))
            if self._reference_date_v2(text) and not self._reference_time_v2(text):
                return self._lookup_reply(text,self._reference_offer_v2(text,flow))
            return self._lookup_reply(text,self._reference_choose_v2(text,flow))
        if flow['stage'] == 'ready':
            if operation and operation != 'details':
                flow['operation'] = operation
                return self._lookup_reply(text,self._reference_begin_action_v2(text,flow))
            # Ordinary questions continue through the existing receptionist.
            return None
        return self._lookup_reply(text,'Please check the appointment reference before making another change.')

    def process_turn(self, patient_text, speaker_assessment=None):
        import re
        text = str(patient_text or '').strip()
        value = self._normalize_confirmation_text(text)
        explicit_handover = bool(re.search(r'\b(?:new speaker|different person speaking|someone else (?:is )?(?:speaking|talking)|i am (?:her|his|their) (?:husband|wife|partner|friend)|i m (?:her|his|their) (?:husband|wife|partner|friend))\b',value))
        change = isinstance(speaker_assessment,dict) and bool(speaker_assessment.get('possible_change') or speaker_assessment.get('returning_voice_candidate'))
        if change or (explicit_handover and not self.session_state.get('speaker_change_pending')):
            self.session_state['speaker_hold_v2'] = None
            self._reference_clear_v2(forget_patient=True)
            assessment = dict(speaker_assessment or {})
            if explicit_handover:
                assessment['possible_change'] = True
            return self._turn_before_reference_v2(text,assessment)
        if self.session_state.get('speaker_change_pending'):
            self._reference_clear_v2(forget_patient=True)
            return self._turn_before_reference_v2(text,speaker_assessment)
        assessment = speaker_assessment if isinstance(speaker_assessment,dict) else {}
        verification = assessment.get('multimodel_verification') or {}
        # Short yes/no turns are often unavailable/uncertain and must not cause
        # false handovers. Sustained distinct-voice evidence or an unresolved
        # independent verification pauses booking writes until resolved.
        sustained = (assessment.get('decision') == 'uncertain' and
                     float(assessment.get('candidate_audio_seconds') or 0) >= 3 and
                     assessment.get('similarity') is not None and
                     float(assessment['similarity']) < 0.30)
        unresolved = (assessment.get('decision') == 'uncertain' and verification and
                      verification.get('status') != 'same_speaker_evidence_false_split_suspected')
        if sustained or unresolved:
            self.session_state['speaker_hold_v2'] = True
            self.session_state['pending_action'] = None
            self.session_state['same_call_name_correction_v1'] = None
            flow = self.session_state.get('reference_flow_v2')
            if flow and flow.get('stage') in {'name','confirm_cancel','confirm_reschedule'}:
                flow['stage'] = 'ready'
            return self._lookup_reply(text,'The voice comparison is unclear, so I have paused appointment changes. Please say a full sentence about your request, or tell me if someone else is speaking now.')
        if self.session_state.get('speaker_hold_v2'):
            if assessment.get('decision') == 'same':
                self.session_state['speaker_hold_v2'] = None
                # Never apply an old yes after the pause. Prepare a fresh action.
                return self._lookup_reply(text,'Thank you. The previous change confirmation was cleared. Please tell me the change you want, so I can read it back for a fresh confirmation.')
            return self._lookup_reply(text,'Appointment changes remain paused while the voice comparison is unclear. Please say a full sentence, or tell me if someone else is speaking.')
        flow = self.session_state.get('reference_flow_v2')
        # Naming another patient invalidates the old reference before any yes.
        if (flow or self.session_state.get('same_call_booking_v1')) and re.search(r'\b(?:someone else|another patient|different patient|for myself|my own appointment)\b',value):
            self._reference_clear_v2(forget_patient=True)
            return self._lookup_reply(text,'Whose appointment would you like to manage? Please provide that patient’s appointment reference.')
        reply = self._reference_turn_v2(text)
        if reply is not None:
            return reply
        return self._turn_before_reference_v2(text,speaker_assessment)



    _reference_turn_before_date_v3 = _reference_turn_v2
    _reference_id_before_date_v3 = _reference_id_v2

    def _reference_id_v2(self, text, bare=False):
        import re
        # Bare spoken numbers are IDs only while the receptionist is asking
        # for a reference. Elsewhere "fourteen" may be a day and "two" an
        # offered option; neither may silently switch to another patient.
        if not bare and not re.search(r'\b(?:appointment|booking|reference|id)\b',self._normalize_confirmation_text(text)):
            return None
        return self._reference_id_before_date_v3(text,bare)

    def _reference_number_text_v3(self, text):
        import re
        value = str(text).lower().replace('-', ' ')
        value = re.sub(r'\ba\s*\.?\s*m\.?\b', 'am', value)
        value = re.sub(r'\bp\s*\.?\s*m\.?\b', 'pm', value)
        value = re.sub(r'[.,;!?]', ' ', value)
        value = re.sub(r'(?<=\d)(?=(?:am|pm)\b)', ' ', value)
        units = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,
                 'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'thirteen':13,
                 'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,
                 'nineteen':19,'twenty':20,'thirty':30,'forty':40,'fifty':50,
                 'sixty':60,'seventy':70,'eighty':80,'ninety':90}
        ordinals = {'first':1,'second':2,'third':3,'fourth':4,'fifth':5,'sixth':6,'seventh':7,
                    'eighth':8,'ninth':9,'tenth':10,'eleventh':11,'twelfth':12,'thirteenth':13,
                    'fourteenth':14,'fifteenth':15,'sixteenth':16,'seventeenth':17,
                    'eighteenth':18,'nineteenth':19,'twentieth':20,'thirtieth':30}
        number_words = '(?:' + '|'.join(units) + ')'
        def numeric(words):
            tokens = words.split()
            if len(tokens) == 1:
                return units.get(tokens[0])
            if len(tokens) == 2 and units.get(tokens[0],0) in (20,30,40,50,60,70,80,90) and 0 < units.get(tokens[1],99) < 10:
                return units[tokens[0]] + units[tokens[1]]
            return None
        # Parse the spoken year before day/time words. Never borrow a year from
        # the stored booking or current clock for reference verification.
        def year(match):
            suffix = match.group(1).strip()
            number = numeric(suffix)
            return str(2000 + number) if number is not None else match.group()
        value = re.sub(r'\btwo thousand(?: and)?\s+('+number_words+r'(?:\s+'+number_words+r')?)\b',year,value)
        value = re.sub(r'\btwenty\s+((?:ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)|twenty(?:\s+(?:one|two|three|four|five|six|seven|eight|nine))?)\b',year,value)
        # Only twenty/thirty + ordinal are combined as a calendar day.
        value = re.sub(r'\b(twenty|thirty)\s+('+'|'.join(ordinals)+r')\b',
                       lambda m:str(units[m[1]]+ordinals[m[2]]) if ordinals[m[2]]<10 else m.group(),value)
        value = re.sub(r'\b('+'|'.join(ordinals)+r')\b',lambda m:str(ordinals[m[1]]),value)
        value = re.sub(r'\b(twenty|thirty|forty|fifty)\s+(one|two|three|four|five|six|seven|eight|nine)\b',
                       lambda m:str(units[m[1]]+units[m[2]]),value)
        value = re.sub(r'\b('+'|'.join(units)+r')\b',lambda m:str(units[m[1]]),value)
        value = re.sub(r'(\d+)(?:st|nd|rd|th)\b',r'\1',value)
        return ' '.join(value.split())

    def _reference_date_parts_v3(self, text):
        import re
        months = {'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,
                  'july':7,'august':8,'september':9,'october':10,'november':11,'december':12}
        months.update({key[:3]:number for key,number in list(months.items())})
        value = self._reference_number_text_v3(text)
        parts = {}
        years = set(re.findall(r'\b(?:19|20|21)\d{2}\b',value))
        if len(years) == 1:
            parts['year'] = int(next(iter(years)))
        found_months = {months[m] for m in re.findall(r'\b('+'|'.join(months)+r')\b',value)}
        if len(found_months) == 1:
            parts['month'] = next(iter(found_months))
        days = set()
        for match in re.finditer(r'\b(?:'+'|'.join(months)+r')\s+(?:the\s+)?(\d{1,2})\b',value):
            if 1<=int(match[1])<=31: days.add(int(match[1]))
        for match in re.finditer(r'\b(\d{1,2})\s+(?:of\s+)?(?:'+'|'.join(months)+r')\b',value):
            if 1<=int(match[1])<=31: days.add(int(match[1]))
        # An ordinal day is explicit even when the month arrives next turn.
        ordinal = re.search(r'\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|thirtieth|\d+(?:st|nd|rd|th))\b',str(text).lower())
        if ordinal:
            converted = self._reference_number_text_v3(ordinal.group())
            # Include a preceding "twenty" or "thirty" if present.
            prefix = str(text).lower()[:ordinal.start()]
            tens = re.search(r'\b(twenty|thirty)\s*$',prefix)
            if tens: converted=self._reference_number_text_v3(tens[1]+' '+ordinal.group())
            if converted.isdigit() and 1<=int(converted)<=31: days.add(int(converted))
        if re.fullmatch(r'(?:the |day |day is )?\d{1,2}',value):
            number=int(value.split()[-1])
            if 1<=number<=31: days.add(number)
        if len(days)==1:
            parts['day']=next(iter(days))
        return parts

    def _reference_date_v2(self, text):
        import re
        from datetime import datetime, timedelta
        raw = str(text).lower()
        dates = set()
        # Keep separators for exact ISO and Australian numeric dates.
        for pattern,fmt in ((r'\b\d{4}-\d{1,2}-\d{1,2}\b','%Y-%m-%d'),
                            (r'\b\d{1,2}/\d{1,2}/\d{4}\b','%d/%m/%Y')):
            for match in re.finditer(pattern,raw):
                try: dates.add(datetime.strptime(match.group(),fmt).date().isoformat())
                except ValueError: return None
        if dates:
            return next(iter(dates)) if len(dates)==1 else None
        parts=self._reference_date_parts_v3(text)
        if all(k in parts for k in ('day','month','year')):
            try: return datetime(parts['year'],parts['month'],parts['day']).date().isoformat()
            except ValueError: return None
        # Relative dates remain useful for new scheduling preferences. The
        # reference-verification wrapper below requires an explicit year.
        if re.search(r'\btomorrow\b',raw): return (sydney_now().date()+timedelta(days=1)).isoformat()
        if re.search(r'\btoday\b',raw): return sydney_now().date().isoformat()
        return None

    def _reference_time_v2(self, text):
        import re
        value=self._reference_number_text_v3(text)
        matches=list(re.finditer(r'\b(\d{1,2})(?::(\d{2})|\s+(\d{2}))?\s*(am|pm)\b',value))
        times=set()
        for match in matches:
            hour,minute=int(match[1]),int(match[2] or match[3] or 0)
            if not 1<=hour<=12 or minute>=60: return None
            times.add('%02d:%02d' % (hour%12+(12 if match[4]=='pm' else 0),minute))
        if matches: return next(iter(times)) if len(times)==1 else None
        # A colon time without a period is retained, then AM/PM is clarified
        # separately when necessary. 00 and 13..23 are unambiguous 24h times.
        matches=list(re.finditer(r'\b(\d{1,2}):(\d{2})\b',value))
        for match in matches:
            if int(match[1])>=24 or int(match[2])>=60: return None
            times.add('%02d:%02d' % (int(match[1]),int(match[2])))
        return next(iter(times)) if len(times)==1 else None

    def _reference_turn_v2(self, text):
        import re
        from datetime import datetime
        flow=self.session_state.get('reference_flow_v2')
        if not flow or flow.get('stage')!='verify_details':
            return self._reference_turn_before_date_v3(text)
        normalized=self._normalize_confirmation_text(text)
        if normalized in {'stop','stop please','never mind','cancel this request','do not update','don t update','goodbye','bye'} or self._reference_id_v2(text) is not None:
            return self._reference_turn_before_date_v3(text)
        if re.search(r'\b(?:give me (?:a |one )?(?:moment|minute)|hold on|just a moment|please wait)\b',normalized):
            return self._lookup_reply(text,'Of course. Take your time; I will keep the booking reference while you check.')
        number_text=self._reference_number_text_v3(text)
        if re.search(r'\b(?:or|maybe|perhaps|not sure|unsure)\b',normalized):
            return self._lookup_reply(text,'Please check and give one definite booked date and time. I will not choose between uncertain details.')
        if len(set(re.findall(r'\b(?:19|20|21)\d{2}\b',number_text)))>1:
            return self._lookup_reply(text,'I heard more than one year. Please tell me the one booked year to use.')
        if self._reference_time_v2(text) is None and re.search(r'\b\d{1,2}(?::\d{2}|\s+\d{2})?\s*(?:am|pm)\b',number_text):
            return self._lookup_reply(text,'I could not verify one valid time from that. Please give one booked time with AM or PM.')
        parts=flow.setdefault('date_parts_v3',{})
        parts.update(self._reference_date_parts_v3(text))
        # An exact full date replaces partial earlier components. A newly
        # stated invalid full date must not fall back to the old stored date.
        date=self._reference_date_v2(text)
        if date and not re.search(r'\b(?:today|tomorrow)\b',normalized):
            flow['given_date']=date
            parsed=datetime.strptime(date,'%Y-%m-%d')
            parts.update(year=parsed.year,month=parsed.month,day=parsed.day)
        elif all(k in parts for k in ('day','month','year')):
            try:
                flow['given_date']=datetime(parts['year'],parts['month'],parts['day']).date().isoformat()
            except ValueError:
                flow.pop('given_date',None); flow['date_parts_v3']={}
                return self._lookup_reply(text,'That is not a valid calendar date. Please give the booked day, month and year again. I have kept the time you supplied.')
        time=self._reference_time_v2(text)
        if time:
            flow['given_time']=time
            hour=int(time[:2])
            has_period=bool(re.search(r'\b(?:am|pm)\b',self._reference_number_text_v3(text)))
            flow['time_period_v3']=has_period or hour==0 or hour>12
        elif normalized in {'am','a m','in the morning','morning','pm','p m','in the afternoon','afternoon','in the evening','evening'} and flow.get('given_time'):
            hour,minute=map(int,flow['given_time'].split(':'))
            pm=normalized in {'pm','p m','in the afternoon','afternoon','in the evening','evening'}
            flow['given_time']='%02d:%02d' % (hour%12+(12 if pm else 0),minute)
            flow['time_period_v3']=True
        if not flow.get('given_date'):
            missing=[label for key,label in (('day','day'),('month','month'),('year','year')) if key not in parts]
            return self._lookup_reply(text,'I still need the booked '+(' and '.join(missing) if missing else 'date')+'. Please tell me '+(' and '.join(missing) if missing else 'the date')+'; you do not need to repeat details already given.')
        if not flow.get('given_time'):
            return self._lookup_reply(text,'Thank you, I have the booked date. What is the booked time, with AM or PM?')
        if not flow.get('time_period_v3'):
            return self._lookup_reply(text,'I have the time as '+flow['given_time']+'. Is that AM or PM?')
        date=flow['given_date']; time=flow['given_time']
        hour,minute=map(int,time.split(':'))
        canonical=date+' at '+str(hour%12 or 12)+':%02d' % minute+(' PM' if hour>=12 else ' AM')
        answer=self._reference_turn_before_date_v3(canonical)
        # Keep genuine caller wording in model history and reason evidence.
        for message in reversed(self.messages):
            if message.get('role')=='user':
                if message.get('content')==canonical: message['content']=text
                break
        self.current_patient_text=text
        if flow.get('stage')=='verify_details' and not flow.get('given_date'):
            flow['date_parts_v3']={}
            flow.pop('time_period_v3',None)
        return answer



    # A single final routing layer: handover, public questions, unfinished
    # new-booking spelling, then existing-booking management. A name correction
    # during collection is never interpreted as a previous-booking lookup.
    _stable_reference_turn_v4 = process_turn
    _stable_dialogue_turn_v4 = _turn_before_reference_v2
    _stable_static_before_v4 = _verified_static_clinic_answer
    _stable_prepare_before_v4 = _prepare_transaction_confirmation
    _stable_name_handle_before_v4 = _nameflow_handle
    _stable_reference_guard_before_v4 = _reference_guard_v2

    def _reference_guard_v2(self, appointment_id):
        if self.session_state.get('phone_result_uncertain_v4'):
            return {'success':False,'reason':'A previous phone request has an unverified outcome. Check the appointment before making another change.'}
        return self._stable_reference_guard_before_v4(appointment_id)

    def _nameflow_handle(self, text, flow):
        answer=self._stable_name_handle_before_v4(text,flow)
        if answer is None and self.session_state.get('verified_patient_name') and self.session_state.get('selected_booking_v4'):
            selected=self.session_state['selected_booking_v4']
            arguments=dict(self.session_state.get('nameflow_booking_arguments') or {})
            arguments.update(patient_name=self.session_state['verified_patient_name'],slot_id=selected['slot_id'])
            result=self.book_appointment(confirmed=False,**arguments)
            if result.get('requires_confirmation'):
                return self._booking_confirmation_text_v1(result['confirmation_details'])
            return str(result.get('reason') or result.get('instruction') or 'Please provide the remaining appointment details.')
        return answer

    def _stable_snapshot_v4(self):
        from copy import deepcopy
        state = self.session_state
        keys = ('active_service','last_available_slots','selected_slot_id',
                'availability_preferences_v1','public_request_v4','selected_booking_v4',
                'nameflow_v1','nameflow_booking_arguments','nameflow_booking_requested')
        draft = {key:deepcopy(state.get(key)) for key in keys}
        pending = state.get('pending_action') or {}
        selected=state.get('selected_booking_v4') or {}
        completed=state.get('same_call_booking_v1') or {}
        unfinished_selection=bool(selected and (not completed or selected.get('slot_id')!=completed.get('slot_id')))
        is_new_draft=bool(state.get('nameflow_v1') or unfinished_selection or
                          state.get('nameflow_booking_arguments') or pending.get('action')=='book')
        name = state.get('verified_patient_name') or state.get('patient_name')
        if name and is_new_draft:
            draft['claimed_name'] = name
        if not is_new_draft:
            draft['selected_booking_v4']=None
            draft['selected_slot_id']=None
            draft['nameflow_v1']=None
            draft['nameflow_booking_arguments']=None
        if pending.get('action') == 'book':
            draft['nameflow_booking_arguments'] = deepcopy(pending.get('arguments'))
        return draft

    def _handover_set_subject_v1(self, subject):
        from copy import deepcopy
        context = dict(self.session_state.get('handover_context_v1') or {})
        previous = context.get('subject')
        self.session_state['speaker_change_pending'] = None
        self.session_state['pending_action'] = None
        if subject == 'original' and previous != 'original':
            draft = getattr(self,'_handover_draft_v4',None)
            if draft:
                # Restore a conversation draft, never a database reference or
                # transaction authorization. Names must be read back again.
                self.session_state.update({k:deepcopy(v) for k,v in draft.items() if k!='claimed_name'})
                claimed = draft.get('claimed_name')
                self.session_state['verified_patient_name'] = None
                self.session_state['verified_patient_name_key'] = None
                self.session_state['observed_spelled_name_confirmed'] = False
                self.session_state['pending_name_verification'] = None
                if claimed:
                    self.session_state['patient_name'] = claimed
                    flow = self.session_state.get('nameflow_v1') or {'parts':claimed.split(),'attempts':0}
                    flow.update(stage='collect', booking=self.session_state.get('nameflow_booking_arguments'))
                    self.session_state['nameflow_v1'] = flow
        elif previous and previous != subject:
            self._reference_clear_v2(forget_patient=True)
            self.session_state['public_request_v4'] = {}
            self.session_state['selected_booking_v4'] = None
        # If this caller just clarified "for me", retain the public availability,
        # date, selected slot and spelling collected SINCE the voice transition.
        # The previous caller's private reference state was cleared at transition.
        context['subject'] = subject
        self.session_state['handover_context_v1'] = context
        self.messages.append({'role':'system','content':
            'Patient subject clarified: '+subject+'. Public scheduling preferences collected in this context remain valid. '
            'A restored draft is not identity verification or a database-write authorization. Require name readback and a fresh exact transaction confirmation.'})

    def _stable_date_v4(self, text):
        import re
        from datetime import timedelta
        date = self._reference_date_v2(text)
        if date:
            return date
        value = self._normalize_confirmation_text(text)
        days = ('monday','tuesday','wednesday','thursday','friday','saturday','sunday')
        found = [i for i,day in enumerate(days) if re.search(r'\b'+day+r'\b',value)]
        if len(found) != 1:
            return None
        today = sydney_now().date()
        offset = (found[0]-today.weekday()) % 7
        if offset == 0 and re.search(r'\bnext\b',value):
            offset = 7
        return (today+timedelta(days=offset)).isoformat()

    def _stable_preferences_v4(self, text):
        import re
        request = dict(self.session_state.get('public_request_v4') or {})
        value = self._normalize_confirmation_text(text)
        for service in service_by_name:
            stem = self._normalize_confirmation_text(service)
            if stem in value or (stem=='dermatology' and re.search(r'\bdermatolog(?:ist|ists|y)\b',value)):
                request['service_name'] = service
        date = self._stable_date_v4(text)
        if date:
            request['date'] = date
        if re.search(r'\bno preferences?\b|\bany consultant\b',value):
            request['consultant_gender'] = 'no_preference'
        self.session_state['public_request_v4'] = request

    def _fast_availability_response(self, tool_result):
        if not isinstance(tool_result,dict) or not tool_result.get('success') or tool_result.get('needs_clarification'):
            return None
        rows = tool_result.get('available_slots') or []
        request = self.session_state.get('public_request_v4') or {}
        date = request.get('date')
        if not rows:
            return 'I found no matching available appointments in that check. Would you like to change the date or consultant preference?'
        if not date:
            date = rows[0]['appointment_date']
        selected = []
        seen = set()
        for row in rows:
            key = (row.get('provider'),row.get('appointment_date'),row.get('appointment_time'))
            if row.get('appointment_date')==date and key not in seen and slot_is_in_future(row):
                seen.add(key); selected.append(row)
                if len(selected)==5: break
        if not selected:
            # A bounded availability list is not proof that a later date has no
            # slots. Let the existing model/tool path request more information.
            return None
        self.session_state['last_available_slots'] = selected
        self.session_state['active_service'] = tool_result.get('service_name') or selected[0]['service_name']
        return ('Available '+str(self.session_state['active_service'])+' appointments on '+selected[0]['display_date']+': '+
                ' '.join('Option '+str(i+1)+': '+s['provider']+' at '+s['display_time']+'.' for i,s in enumerate(selected))+
                ' Which option would you prefer?')

    def _stable_fact_v4(self, text):
        import re
        value = self._normalize_confirmation_text(text)
        selected = []
        if re.search(r'\b(?:cafeteria|cafe|canteen)\b',value): selected.append(('Public Cafeteria','cafeteria'))
        if re.search(r'\b(?:pray|prayer|worship|multi faith)\b',value): selected.append(('Prayer Area','prayer area'))
        if re.search(r'\b(?:playground|play area|space to play|children to play|kids to play)\b',value): selected.append(('Children\'s Play Area','children\'s play area'))
        if re.search(r'\b(?:open|opening|close|closing|hours)\b',value) and (not selected or 'clinic' in value): selected.append(('Clinic Opening Hours','clinic opening hours'))
        if not selected:
            return None
        rows = [row for row in knowledge_rows if row.get('active',True) is True]
        answers=[]
        for title,label in selected:
            contents=list(dict.fromkeys(str(row.get('content') or '').strip() for row in rows
                if self._normalize_confirmation_text(row.get('title',''))==self._normalize_confirmation_text(title) and str(row.get('content') or '').strip()))
            if len(contents)==1:
                answer=contents[0]
                if title=="Children's Play Area":
                    old=[str(row.get('content') or '') for row in rows if self._normalize_confirmation_text(row.get('title',''))=='children s playground']
                    if any(re.search(r'\b(?:no|not)\b',s.lower()) and 'play area' in s.lower() for s in old):
                        answer += ' Another active record says there is no play area, so these records conflict and the facility needs confirmation.'
                answers.append(answer)
            elif len(contents)>1:
                answers.append('The active '+label+' records conflict, so I cannot give one confirmed answer.')
            else:
                answers.append('The current records do not specify the '+label+'.')
        return ' '.join(dict.fromkeys(answers))

    def _verified_static_clinic_answer(self, text):
        return self._stable_fact_v4(text) or self._stable_static_before_v4(text)

    def _prepare_transaction_confirmation(self, action, arguments, details, confirmed):
        if self.session_state.get('phone_result_uncertain_v4'):
            return {'success':False,'reason':'A previous phone request has an unverified outcome. Check the booking before making another change; I will not repeat it automatically.'}
        if self.session_state.get('speaker_hold_v2'):
            return {'success':False,'reason':'Voice clarification is pending. No appointment write is authorized.'}
        return self._stable_prepare_before_v4(action,arguments,details,confirmed)

    def _stable_pick_slot_v4(self, text):
        import re
        value=self._normalize_confirmation_text(text)
        slots=self.session_state.get('last_available_slots') or []
        if not slots:
            return None
        # Selection phrases, not an unrelated mention of time in a question.
        if not re.search(r'\b(?:take|choose|select|prefer|book|perfect|fine|option|first|second|third|fourth|fifth)\b',value):
            return None
        ordinal={'first':1,'second':2,'third':3,'fourth':4,'fifth':5,'one':1,'two':2,'three':3,'four':4,'five':5}
        match=re.fullmatch(r'(?:option |the )?(first|second|third|fourth|fifth|one|two|three|four|five|[1-5])(?: one| option| please)?',value)
        if match:
            index=ordinal.get(match[1]) or int(match[1])
            return slots[index-1] if index<=len(slots) else None
        time=self._reference_time_v2(text)
        date=self._stable_date_v4(text)
        matches=[s for s in slots if time and str(s.get('appointment_time',''))[:5]==time and (not date or s['appointment_date']==date)]
        if len(matches)>1:
            named=[s for s in matches if any(len(word)>2 and word not in {'doctor','dr'} and re.search(r'\b'+re.escape(word)+r'\b',value)
                for word in self._normalize_confirmation_text(s.get('provider','')).split())]
            matches=named
        return matches[0] if len(matches)==1 else None

    def _stable_prepare_selected_v4(self, text):
        selected=self.session_state.get('selected_booking_v4')
        if not selected:
            return None
        gate=self._handover_subject_gate_v1()
        if gate:
            return self._lookup_reply(text,gate['reason'])
        name=self.session_state.get('verified_patient_name')
        arguments=dict(self.session_state.get('nameflow_booking_arguments') or {})
        arguments.update(patient_name=name or self.session_state.get('patient_name') or '',slot_id=selected['slot_id'])
        # Reuse real availability/name/confirmation gates and atomic booking RPC.
        result=self.book_appointment(confirmed=False,**arguments)
        if result.get('requires_confirmation'):
            return self._lookup_reply(text,self._booking_confirmation_text_v1(result['confirmation_details']))
        if result.get('requires_name_spelling') or self.session_state.get('nameflow_v1'):
            return self._lookup_reply(text,self._nameflow_prompt())
        return self._lookup_reply(text,str(result.get('reason') or result.get('instruction') or 'Please provide the remaining appointment details.'))

    def process_turn(self, patient_text, speaker_assessment=None):
        import re
        from copy import deepcopy
        text=str(patient_text or '').strip()
        value=self._normalize_confirmation_text(text)
        assessment=dict(speaker_assessment or {}) if isinstance(speaker_assessment,dict) else {}
        fresh=bool(assessment.get('possible_change') or assessment.get('returning_voice_candidate'))
        announced=bool(re.search(r'\b(?:i am (?:a )?different person|new speaker|someone else is speaking|i am taking over)\b',value))
        if not self.session_state.get('speaker_change_pending') and not self.session_state.get('handover_context_v1') and self._handover_relationship_v1(text) and self.session_state.get('reference_verified_v2'):
            announced=True
        if fresh or announced:
            self._handover_draft_v4=self._stable_snapshot_v4()
            self.session_state['public_request_v4']={}
            self.session_state['selected_booking_v4']=None
            if announced: assessment['possible_change']=True
            return self._stable_reference_turn_v4(text,assessment)
        verification=assessment.get('multimodel_verification') or {}
        independent_same=verification.get('status')=='same_speaker_evidence_false_split_suspected'
        if independent_same:
            # Follow the existing verifiers' explicit false-split verdict.
            # Never invent a same-speaker decision from borderline scores.
            assessment=dict(assessment,decision='same',candidate_audio_seconds=0,multimodel_verification={})
        write_flow=self.session_state.get('same_call_name_correction_v1') or {}
        reference_flow=self.session_state.get('reference_flow_v2') or {}
        mutation_pending=bool(self.session_state.get('pending_action') or write_flow.get('stage')=='confirm' or reference_flow.get('stage') in {'confirm_cancel','confirm_reschedule'})
        if (not mutation_pending and not reference_flow and not self.session_state.get('speaker_hold_v2') and
                verification.get('status')=='uncertain_do_not_trigger_change' and
                not re.search(r'\b(?:previous|existing|reschedule|cancel|booking reference|appointment id)\b',value)):
            # An inconclusive split does not stop public enquiries or collection
            # of a NEW booking draft. No write is possible from this state; final
            # transaction approval still uses the normal voice safety gate.
            assessment=dict(assessment,candidate_audio_seconds=0,multimodel_verification={})
        pending=self.session_state.get('speaker_change_pending')
        if pending:
            relationship=self._handover_relationship_v1(text)
            subject=self._handover_subject_v1(text)
            if not subject and re.search(r'\b(?:for my friend|on behalf of (?:him|her|them))\b',value): subject='original'
            if not relationship and not subject:
                return self._lookup_reply(text,'May I ask your relationship to the patient, and whether this request is for them or for you?')
            self.session_state['speaker_change_pending']=None
            self.session_state['pending_action']=None
            self.session_state['handover_context_v1']={'relationship':relationship,'subject':None}
            if subject:
                self._handover_set_subject_v1(subject)
            self._stable_preferences_v4(text)
            # Preserve the actual scheduling request contained in the same
            # utterance as the relationship answer instead of throwing it away.
            if re.search(r'\b(?:available|dermatolog(?:ist|y)|book|appointment|tomorrow|tuesday)\b',value):
                return self._stable_dialogue_turn_v4(text,assessment)
            restored=self.session_state.get('nameflow_v1')
            if subject=='original' and restored and restored.get('parts'):
                return self._lookup_reply(text,'We can continue the patient\'s draft. '+self._nameflow_readback(restored))
            return self._lookup_reply(text,'Thank you. Is the appointment for the original patient or for you?' if not subject else 'Thank you. What would you like to do next?')
        context=self.session_state.get('handover_context_v1')
        if context and not context.get('subject'):
            subject=self._handover_subject_v1(text)
            if re.search(r'\bmy name\b',value): subject='self'
            if re.search(r'\b(?:for my friend|on behalf of (?:him|her|them))\b',value): subject='original'
            if subject:
                self._handover_set_subject_v1(subject)
                if value in {'for me','for me yes','for myself','for the original patient','for him','for her','for them'}:
                    selected_reply=self._stable_prepare_selected_v4(text)
                    if selected_reply is not None: return selected_reply
                    return self._lookup_reply(text,'Thank you. Please continue your request.')
        elif context and context.get('subject') and value in {'for me','for me yes','for myself','for the original patient','for him','for her','for them'}:
            # Subject clarification is not assent to spelling or a transaction.
            return self._lookup_reply(text,'Thank you. I have kept the current request. Please confirm the exact name or appointment details I read back separately.')
        # Process uncertain-write holds before any shortcut that could prepare or
        # execute a transaction. Normal short yes/no evidence does not create one.
        if self.session_state.get('speaker_hold_v2') or (assessment.get('decision')=='uncertain' and
                ((float(assessment.get('candidate_audio_seconds') or 0)>=3 and float(assessment.get('similarity',1))<0.30) or assessment.get('multimodel_verification'))):
            return self._stable_reference_turn_v4(text,assessment)
        fact=self._stable_fact_v4(text)
        if fact:
            return self._lookup_reply(text,fact)
        flow=self.session_state.get('nameflow_v1')
        reference=self.session_state.get('reference_flow_v2')
        explicit_existing=bool(re.search(r'\b(?:previous|existing|already booked|booking reference|appointment id|reschedule|cancel my appointment)\b',value))
        new_booking=bool(re.search(r'\b(?:first time|new booking|new appointment|no appointment id|no booking reference|do not have.*reference|don t have.*reference)\b',value))
        if not reference and not explicit_existing:
            self._stable_preferences_v4(text)
        if flow and not explicit_existing:
            return self._stable_dialogue_turn_v4(text,assessment)
        last_answer=next((str(m.get('content') or '') for m in reversed(self.messages) if m.get('role')=='assistant'),'').lower()
        if re.search(r'\bno reference\b',value) and ('preference' in last_answer or (self.session_state.get('public_request_v4') or {}).get('service_name')):
            self.session_state['consultant_preference_check_v4']=True
            return self._lookup_reply(text,'Do you mean no preference for the consultant? This is a new booking, so an appointment ID is not needed.')
        if self.session_state.get('consultant_preference_check_v4'):
            if self._correction_assent_v1(text) or re.search(r'\bno preference\b',value):
                self.session_state['consultant_preference_check_v4']=None
                self.session_state['reference_flow_v2']=None
                self.session_state['reference_verified_v2']=None
                self.session_state['public_request_v4']['consultant_gender']='no_preference'
                return self._stable_dialogue_turn_v4('No preference for the consultant.',assessment)
            self.session_state['consultant_preference_check_v4']=None
        if new_booking:
            self.session_state['reference_flow_v2']=None
            self.session_state['reference_verified_v2']=None
            self.session_state['lookup_nameflow']=None
            self.session_state['lookup_operation']=None
            reference=None
        # Remember proactive spelling once, including on-behalf-of spelling,
        # even when a consultant/date question was the last question asked.
        groups=self._nameflow_groups(text)
        if len(groups)>=2 and re.search(r'\b(?:name|his name|her name|full name)\b',value) and not explicit_existing and not reference:
            gate=self._handover_subject_gate_v1()
            if gate: return self._lookup_reply(text,gate['reason'])
            flow={'stage':'collect','parts':groups,'attempts':0,'booking':self.session_state.get('nameflow_booking_arguments')}
            self.session_state['nameflow_v1']=flow
            return self._lookup_reply(text,self._nameflow_readback(flow))
        picked=self._stable_pick_slot_v4(text) if not reference and not explicit_existing else None
        if picked:
            self.session_state['selected_booking_v4']=dict(picked)
            self.session_state['selected_slot_id']=picked['slot_id']
            return self._stable_prepare_selected_v4(text)
        operation=self._reference_operation_v2(text)
        management=bool(reference or explicit_existing or
            (operation=='name' and not flow and not self.session_state.get('selected_booking_v4')) or
            (self.session_state.get('same_call_booking_v1') and operation in {'name','cancel','reschedule'}))
        if management and not new_booking:
            return self._stable_reference_turn_v4(text,assessment)
        if self.session_state.get('pending_action'):
            result=self._confirmed_booking_turn_v1(text)
            if result is not None: return result
        return self._stable_dialogue_turn_v4(text,assessment)


# ============================================================
# FACTORY FUNCTION
# ============================================================

def create_receptionist_session():

    return ReceptionistSession()

