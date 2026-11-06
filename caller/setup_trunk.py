"""One-time setup: register the Twilio Elastic SIP trunk as a LiveKit outbound trunk.

Usage: python -m caller.setup_trunk
Prints the trunk ID and writes it to .env as LIVEKIT_SIP_TRUNK_ID.
"""

from __future__ import annotations

import asyncio

from livekit import api

from .config import ROOT, require


async def main() -> None:
    trunk = api.SIPOutboundTrunkInfo(
        name="patient-caller-twilio",
        address=require("TWILIO_TERMINATION_URI").removeprefix("sip:"),
        numbers=[require("CALLER_ID_NUMBER")],
        auth_username=require("TWILIO_SIP_USERNAME"),
        auth_password=require("TWILIO_SIP_PASSWORD"),
    )

    lkapi = api.LiveKitAPI()
    try:
        existing = await lkapi.sip.list_outbound_trunk(api.ListSIPOutboundTrunkRequest())
        for t in existing.items:
            if t.name == trunk.name:
                print(f"Outbound trunk already exists: {t.sip_trunk_id}")
                _save_trunk_id(t.sip_trunk_id)
                return

        created = await lkapi.sip.create_outbound_trunk(api.CreateSIPOutboundTrunkRequest(trunk=trunk))
        print(f"Created outbound trunk: {created.sip_trunk_id}")
        _save_trunk_id(created.sip_trunk_id)
    finally:
        await lkapi.aclose()


def _save_trunk_id(trunk_id: str) -> None:
    env_path = ROOT / ".env"
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    lines = [line for line in lines if not line.startswith("LIVEKIT_SIP_TRUNK_ID=")]
    lines.append(f"LIVEKIT_SIP_TRUNK_ID={trunk_id}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved LIVEKIT_SIP_TRUNK_ID to .env")


if __name__ == "__main__":
    asyncio.run(main())
