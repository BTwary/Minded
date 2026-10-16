const RESET_STARTED_EVENT = "minded:workspace-reset-started";
const RESET_COMPLETED_EVENT = "minded:workspace-reset-completed";
const RESET_FAILED_EVENT = "minded:workspace-reset-failed";
const UPLOAD_STARTED_EVENT = "minded:workspace-upload-started";
const UPLOAD_FINISHED_EVENT = "minded:workspace-upload-finished";

let workspaceRequestGeneration = 0;

export function getWorkspaceRequestGeneration(): number {
  return workspaceRequestGeneration;
}

export function beginWorkspaceReset(): number {
  workspaceRequestGeneration += 1;
  if (typeof window !== "undefined") {
    window.dispatchEvent(new CustomEvent(RESET_STARTED_EVENT, {
      detail: { generation: workspaceRequestGeneration },
    }));
  }
  return workspaceRequestGeneration;
}

function clearActiveWorkspaceSelection(): unknown | null {
  if (typeof window === "undefined") return null;
  try {
    window.localStorage.removeItem("minded-active-dataset");
    window.localStorage.removeItem("minded-active-dataset-id");
    return null;
  } catch (error) {
    console.error("Unable to clear persisted active dataset selection.", error);
    return error;
  }
}

export function completeWorkspaceReset(): void {
  workspaceRequestGeneration += 1;
  const selectionError = clearActiveWorkspaceSelection();
  if (typeof window !== "undefined") {
    window.dispatchEvent(new CustomEvent(RESET_COMPLETED_EVENT, {
      detail: { generation: workspaceRequestGeneration },
    }));
  }
  if (selectionError) {
    throw new Error("The backend reset completed, but persisted active dataset selection could not be invalidated.");
  }
}

export function failWorkspaceReset(): void {
  workspaceRequestGeneration += 1;
  clearActiveWorkspaceSelection();
  if (typeof window !== "undefined") {
    window.dispatchEvent(new CustomEvent(RESET_FAILED_EVENT, {
      detail: { generation: workspaceRequestGeneration },
    }));
  }
}

export function beginWorkspaceUpload(): void {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new CustomEvent(UPLOAD_STARTED_EVENT));
  }
}

export function finishWorkspaceUpload(): void {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new CustomEvent(UPLOAD_FINISHED_EVENT));
  }
}

export const workspaceResetEvents = {
  started: RESET_STARTED_EVENT,
  completed: RESET_COMPLETED_EVENT,
  failed: RESET_FAILED_EVENT,
  uploadStarted: UPLOAD_STARTED_EVENT,
  uploadFinished: UPLOAD_FINISHED_EVENT,
};
