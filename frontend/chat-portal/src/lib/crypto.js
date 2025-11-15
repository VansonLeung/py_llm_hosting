export async function hashSecret(value) {
  if (!value) return ""
  if (window?.crypto?.subtle) {
    const encoder = new TextEncoder()
    const data = encoder.encode(value)
    const hashBuffer = await window.crypto.subtle.digest("SHA-256", data)
    return Array.from(new Uint8Array(hashBuffer))
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join("")
  }
  return btoa(value)
}

export function maskSecret(value = "", visibleChars = 4) {
  if (!value) return ""
  const visible = value.slice(-visibleChars)
  return `${"*".repeat(Math.max(0, value.length - visibleChars))}${visible}`
}
